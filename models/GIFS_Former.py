import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from models.VertexFormer import DeformableTransformerDecoderLayer as VertexDecoderLayer
from models.VertexFormer import DeformableTransformerDecoder as VertexDecoder
from models.mlp import MLP
from models.deformable_transformer import DeformableTransformerEncoderLayer, DeformableTransformerEncoder, \
    DeformableTransformerDecoder, DeformableTransformerDecoderLayer, DeformableAttnDecoderLayer
from models.ops.modules import MSDeformAttn
from models.resnet import ResNetBackbone
from models.corner_models import PositionEmbeddingSine
from utils.misc import NestedTensor
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from models.cross_attention import PreNorm, Attention, FeedForward


class GIFS_Former(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GIFS_Former, self).__init__()
        self.backbone = ResNetBackbone()
        backbone_strides = self.backbone.strides
        backbone_num_channel = self.backbone.num_channels

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_feature_levels = 4

        input_project_list = []  # project backbone feature maps
        for _ in range(len(backbone_strides)):
            # in_channels: backbone output channels
            in_channels = backbone_num_channel[_]
            input_project_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim)
            ))
        for _ in range(self.num_feature_levels - len(backbone_strides)):
            input_project_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, hidden_dim)
            ))
            in_channels = hidden_dim

        self.input_project = nn.ModuleList(input_project_list)
        self.img_position_embedding = PositionEmbeddingSine(hidden_dim // 2)

        self.edge_input_fc = nn.Linear(input_dim * 2, hidden_dim)
        # define grid of feature map
        self.patch_size = 8
        patch_dim = (self.patch_size ** 2) * self.input_dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
        )
        self.transformer = GIFSTransformer()
        self.coords_embed = MLP(self.hidden_dim, self.hidden_dim, 2, 3)
        self.class_embed = nn.Linear(self.hidden_dim, 1)
        self.with_poly_refine = True

        # init_transformer_params

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(1) * bias_value
        nn.init.constant_(self.coords_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.coords_embed.layers[-1].bias.data, 0)

        for proj in self.input_project:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = self.transformer.vertex_decoder.num_layers
        if self.with_poly_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.coords_embed = _get_clones(self.coords_embed, num_pred)
            nn.init.constant_(self.coords_embed[0].layers[-1].bias.data[2:], -2.0)
        self.transformer.vertex_decoder.coords_embed = self.coords_embed
        self.transformer.vertex_decoder.class_embed = self.class_embed
        # 定义一个transformer

    def get_ms_feat(self, xs, img_mask):
        out: Dict[str, NestedTensor] = {}
        for name, x in sorted(xs.items()):
            m = img_mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

    @staticmethod
    def get_decoder_reference_points(height, width, device):
        # ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
        #                               torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device))
        ref_x, ref_y = torch.meshgrid(torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                                      torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / height
        ref_x = ref_x.reshape(-1)[None] / width
        ref = torch.stack((ref_x, ref_y), -1)
        return ref

    def forward(self, inputs, pixel_features, edge_coords):
        image_feats, feat_mask, all_image_feats = self.backbone(inputs)
        features = self.get_ms_feat(image_feats, feat_mask)
        new_features = list()
        for name, x in sorted(features.items()):
            new_features.append(x)
        features = new_features
        input_features = []
        masks = []
        features_pos = []
        for level, feat in enumerate(features):
            src, mask = feat.decompose()
            mask = mask.to(src.device)
            input_features.append(self.input_project[level](src))
            pos = self.img_position_embedding(src).to(src.dtype)
            features_pos.append(pos)
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(input_features):
            for level in range(len(input_features), self.num_feature_levels):
                if level == len(input_features):
                    proj_feat = self.input_project[level](features[-1].tensors)
                else:
                    proj_feat = self.input_project[level](input_features[-1])
                m = feat_mask
                mask = F.interpolate(m[None].float(), size=proj_feat.shape[-2:]).to(torch.bool)[0].to(proj_feat.device)
                masks.append(mask)
                input_features.append(proj_feat)
                feat_pos = self.img_position_embedding(proj_feat).to(proj_feat.dtype)
                features_pos.append(feat_pos)
        batch_size = edge_coords.size(0)
        edges_num = edge_coords.size(1)
        # handle edge positional encoding
        edge_positional_feat = list()
        coords = edge_coords.long()
        coords = coords.view(batch_size, edges_num, 2, 2)
        for b_i in range(batch_size):
            positional_feat = pixel_features[b_i, coords[b_i, :, :, 1], coords[b_i, :, :, 0], :]
            edge_positional_feat.append(positional_feat)
        edge_positional_feat = torch.stack(edge_positional_feat, dim=0)
        edge_positional_feat = edge_positional_feat.view(batch_size, edges_num, -1)
        edge_feat = self.edge_input_fc(edge_positional_feat.view(batch_size * edges_num, -1))
        edge_feat = edge_feat.view(batch_size, edges_num, -1)
        edge_center = (edge_coords[:, :, 0:2].float() + edge_coords[:, :, 2:4].float()) / 2
        edge_center = edge_center / feat_mask.shape[1]

        # handle vertex map
        grid_feats = self.to_patch_embedding(pixel_features)
        # 像素 independent ---- 像素dependent
        H_tgt = W_tgt = int(np.sqrt(grid_feats.shape[1]))
        grid_ref = self.get_decoder_reference_points(H_tgt, W_tgt, grid_feats.device)
        gifs, inter_references, inter_classes = self.transformer(input_features, masks, features_pos, edge_feat,
                                                                 edge_center, grid_feats, grid_ref)
        # gifs = self.transformer(input_features, masks, features_pos, edge_feat,
        #                         edge_center, grid_feats, grid_ref)
        # return gifs, inter_references, inter_classes
        return gifs


# TODO: 写一个learnable transformer
class LearnableFormer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_queries):
        super(LearnableFormer, self).__init__()
        self.backbone = ResNetBackbone()
        self.num_queries = num_queries
        backbone_strides = self.backbone.strides
        backbone_num_channel = self.backbone.num_channels
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_feature_levels = 4
        input_project_list = []  # project backbone feature maps
        for _ in range(len(backbone_strides)):
            # in_channels: backbone output channels
            in_channels = backbone_num_channel[_]
            input_project_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim)
            ))
        for _ in range(self.num_feature_levels - len(backbone_strides)):
            input_project_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, hidden_dim)
            ))
            in_channels = hidden_dim
        self.img_position_embedding = PositionEmbeddingSine(hidden_dim // 2)
        self.input_project = nn.ModuleList(input_project_list)
        self.img_pos = PositionEmbeddingSine(hidden_dim // 2)
        self.query_embed = nn.Embedding(num_queries, 2)
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.edge_input_mlp = nn.Linear(self.input_dim * 2, hidden_dim)
        self.transformer = LearnableTransFormer(d_model=hidden_dim)
        num_pred = self.transformer.decoder.num_layers
        self.coords_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.class_embed = nn.Linear(hidden_dim, 1)
        self.coords_embed = _get_clones(self.coords_embed, num_pred)
        self.class_embed = _get_clones(self.class_embed, num_pred)
        self.transformer.decoder.coords_embed = self.coords_embed
        self.transformer.decoder.class_embed = self.class_embed

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(hidden_dim, Attention(hidden_dim, hidden_dim, heads=1, dim_head=hidden_dim),
                    context_dim=hidden_dim),
            PreNorm(hidden_dim, FeedForward(hidden_dim))
        ])
        self.patch_size = 8
        patch_dim = (self.patch_size ** 2) * self.input_dim
        self.patch_embedding = nn.Sequential(
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
        )
        self.to_outputs = nn.Linear(hidden_dim, 1)

    def get_ms_feat(self, xs, img_mask):
        out: Dict[str, NestedTensor] = {}
        for name, x in sorted(xs.items()):
            m = img_mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

    @staticmethod
    def get_decoder_reference_points(height, width, device):
        # ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
        #                               torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device))
        ref_x, ref_y = torch.meshgrid(torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                                      torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / height
        ref_x = ref_x.reshape(-1)[None] / width
        ref = torch.stack((ref_x, ref_y), -1)
        return ref

    def forward(self, inputs, pixel_features, edge_coords):
        # edge_coords: positional embedding
        image_feats, feat_mask, all_image_feats = self.backbone(inputs)
        features = self.get_ms_feat(image_feats, feat_mask)
        new_features = list()
        for name, x in sorted(features.items()):
            new_features.append(x)
        features = new_features
        input_features = []
        masks = []
        features_pos = []
        for level, feat in enumerate(features):
            src, mask = feat.decompose()
            mask = mask.to(src.device)
            input_features.append(self.input_project[level](src))
            pos = self.img_position_embedding(src).to(src.dtype)
            features_pos.append(pos)
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(input_features):
            for level in range(len(input_features), self.num_feature_levels):
                if level == len(input_features):
                    proj_feat = self.input_project[level](features[-1].tensors)
                else:
                    proj_feat = self.input_project[level](input_features[-1])
                m = feat_mask
                mask = F.interpolate(m[None].float(), size=proj_feat.shape[-2:]).to(torch.bool)[0].to(proj_feat.device)
                masks.append(mask)
                input_features.append(proj_feat)
                feat_pos = self.img_position_embedding(proj_feat).to(proj_feat.dtype)
                features_pos.append(feat_pos)
        batch_size = edge_coords.size(0)
        edges_num = edge_coords.size(1)
        # handle edge positional encoding
        edge_positional_feat = list()
        coords = edge_coords.long()
        coords = coords.view(batch_size, edges_num, 2, 2)
        for b_i in range(batch_size):
            positional_feat = pixel_features[b_i, coords[b_i, :, :, 1], coords[b_i, :, :, 0], :]
            edge_positional_feat.append(positional_feat)
        edge_positional_feat = torch.stack(edge_positional_feat, dim=0)
        edge_positional_feat = edge_positional_feat.view(batch_size, edges_num, -1)
        edge_feat = self.edge_input_mlp(edge_positional_feat.view(batch_size * edges_num, -1))
        edge_feat = edge_feat.view(batch_size, edges_num, -1)

        # edge_center = (edge_coords[:, :, 0:2].float() + edge_coords[:, :, 2:4].float()) / 2
        # edge_center = edge_center / feat_mask.shape[1]

        query_embeds = self.query_embed.weight
        tgt_embeds = self.tgt_embed.weight
        # query_embeds没用上
        # learnable query --- edge_feat
        learnable_embeds, learnable_reference, learnable_classes = self.transformer(input_features, masks, features_pos, query_embeds, tgt_embeds)
        # edge_feat --- interaction with learnable_embeds
        # cross_attention between edge_feat and learnable feat
        # cross_attention --- deformable cross attention?
        # directly: cross attention
        cross_attn, cross_ff = self.cross_attend_blocks
        # point embedding
        x = cross_attn(edge_feat, context=learnable_embeds, mask=None)
        x = cross_ff(x) + x
        outputs = self.to_outputs(x)
        # grid embedding
        # grid_feats = self.patch_embedding(pixel_features)
        # H_tgt = W_tgt = int(np.sqrt(grid_feats.shape[1]))
        # grid_ref = self.get_decoder_reference_points(H_tgt, W_tgt, grid_feats.device)
        # patch_ca, patch_cf = self.patch_ca
        # point embedding
        # x = patch_ca(grid_feats, context=learnable_embeds, mask=None)
        # x = patch_cf(x) + x
        return outputs, learnable_reference, learnable_classes


# query: learnable query: how to interact? --- pairs cross attention with several query
class LearnableTransFormer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024,
                 dropout=0.1, activation="relu", return_intermediate_dec=False, num_feature_levels=4,
                 dec_n_points=4, enc_n_points=4, query_pos_type="sine"):
        super().__init__()
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        # 定义decoder
        # self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers,
        #                                             return_intermediate_dec)
        self.decoder = VertexDecoder(decoder_layer, num_decoder_layers, True, return_intermediate_dec, False,
                                     query_pos_type)
        self.decoder.pos_trans = nn.Linear(d_model, d_model)
        self.decoder.pos_trans_norm = nn.LayerNorm(d_model)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None, tgt=None, tgt_masks=None):
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                              mask_flatten)
        # prepare input for decoder
        bs, _, c = memory.shape

        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        # 拓展成batch size \times -1
        reference_points = query_embed.sigmoid()
        # reference_points = query_embed
        init_reference_out = reference_points
        hs, inter_reference, point_classes = self.decoder(tgt, reference_points, memory, src_flatten, spatial_shapes,
                                                          level_start_index,
                                                          valid_ratios, query_embed, mask_flatten, tgt_masks)
        return hs, inter_reference, point_classes


class GIFSTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=1, dim_feedforward=1024,
                 dropout=0.1, activation="relu", return_intermediate_dec=False, poly_refine=True, num_feature_levels=4,
                 dec_n_points=4, enc_n_points=4, query_pos_type="sine"):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        # encoder提取特征
        decoder_attn_layer = DeformableAttnDecoderLayer(d_model, dim_feedforward,
                                                        dropout, activation,
                                                        num_feature_levels, nhead, dec_n_points)
        self.interpolate_edge = DeformableTransformerDecoder(decoder_attn_layer, 1, False, with_sa=False)
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.gifs_decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers,
                                                         return_intermediate_dec, with_sa=True)
        self.gifs_fc = MLP(input_dim=d_model, hidden_dim=d_model // 2, output_dim=1, num_layers=2)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.vertex_decoder = VertexDecoder(decoder_layer, num_decoder_layers, poly_refine, return_intermediate_dec,
                                            False, query_pos_type)
        # self.query_embed = nn.Embedding(32 * 32, 2)
        # self.tgt_embed = nn.Embedding(32 * 32, d_model)

        if query_pos_type == 'sine':
            self.vertex_decoder.pos_trans = nn.Linear(d_model, d_model)
            self.vertex_decoder.pos_trans_norm = nn.LayerNorm(d_model)
        #
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, edge_feats, edge_ref, grid_feats, grid_ref):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                              mask_flatten)

        # TODO: queries: 改成可学习的
        # grid_decoder
        bs, _, c = memory.shape
        # memory: f_i
        # grid_ref = grid_ref.expand(bs, -1, -1)
        # init_reference_out = grid_ref
        # hs, inter_references, inter_classes = self.vertex_decoder(grid_feats, grid_ref, memory, src_flatten,
        #                                                           spatial_shapes, level_start_index, valid_ratios,
        #                                                           grid_ref, mask_flatten, None)
        # # # learnable query
        # query_embeds = self.query_embed.weight
        # tgt_embeds = self.tgt_embed.weight
        # query_embeds = query_embeds.unsqueeze(0).expand(bs, -1, -1)
        # tgt = tgt_embeds.unsqueeze(0).expand(bs, -1, -1)
        # reference_points = query_embeds.sigmoid()
        # hs, inter_references, inter_classes = self.vertex_decoder(tgt, reference_points, memory, src_flatten,
        #                                                           spatial_shapes, level_start_index, valid_ratios,
        #                                                           query_embeds, mask_flatten, None)

        # gifs_decoder
        # f_img_edge, _ = self.interpolate_edge(edge_feats, edge_ref, memory, spatial_shapes, level_start_index,
        #                                       valid_ratios, edge_feats, mask_flatten)
        # gifs, _ = self.gifs_decoder(f_img_edge, edge_ref, memory, spatial_shapes, level_start_index, valid_ratios,
        #                             edge_feats, mask_flatten)
        gifs, _ = self.gifs_decoder(edge_feats, edge_ref, memory, spatial_shapes, level_start_index, valid_ratios,
                                    edge_feats, mask_flatten)
        gifs = self.gifs_fc(gifs)

        return gifs
        # return gifs, inter_references, inter_classes


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
