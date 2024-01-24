from functools import wraps
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from models.deformable_transformer import DeformableTransformerEncoderLayer, DeformableTransformerEncoder, \
    DeformableTransformerDecoder, DeformableTransformerDecoderLayer, DeformableAttnDecoderLayer
from models.VertexFormer import DeformableTransformerDecoderLayer as VertexDecoderLayer
from models.VertexFormer import DeformableTransformerDecoder as VertexDecoder
from models.mlp import MLP
from models.ops.modules import MSDeformAttn
from models.resnet import ResNetBackbone
from models.corner_models import PositionEmbeddingSine
from utils.misc import NestedTensor
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from models.cross_attention import PreNorm, Attention, FeedForward


def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn


class Learnable_Former(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(Learnable_Former, self).__init__()
        # define backbone and the projection
        self.backbone = ResNetBackbone()
        backbone_strides = self.backbone.strides
        backbone_num_channel = self.backbone.num_channels
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_feature_levels = 4
        input_proj_list = []  # project backbone feature maps
        for _ in range(len(backbone_strides)):
            # in_channels: backbone output channels
            in_channels = backbone_num_channel[_]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim)
            ))
        for _ in range(self.num_feature_levels - len(backbone_strides)):
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, hidden_dim)
            ))
            in_channels = hidden_dim
        self.input_proj = nn.ModuleList(input_proj_list)
        self.img_pos = PositionEmbeddingSine(hidden_dim // 2)
        self.edge_input_fc = nn.Linear(input_dim * 2, hidden_dim)
        # define grid of feature map
        self.patch_size = 8
        patch_dim = (self.patch_size ** 2) * self.input_dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
        )
        self.num_queries = 500
        self.query_embed = nn.Embedding(self.num_queries, 2)
        self.learnable_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.transformer = ImplicitTransformer(d_model=hidden_dim)

    @staticmethod
    def get_ms_feat(xs, img_mask):
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

    # inputs: input images
    # pixel_features: positional_encoding of each pixel
    # pairs_coords: coordinate of each pairs of point
    def forward(self, inputs, pixel_features, pairs_coords):
        image_feats, feat_mask, all_image_feats = self.backbone(inputs)
        features = self.get_ms_feat(image_feats, feat_mask)
        srcs = []
        masks = []
        all_pos = []
        new_features = list()
        for name, x in sorted(features.items()):
            new_features.append(x)
        features = new_features
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            mask = mask.to(src.device)
            srcs.append(self.input_proj[l](src))
            pos = self.img_pos(src).to(src.dtype)
            all_pos.append(pos)
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = feat_mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0].to(src.device)
                pos_l = self.img_pos(src).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                all_pos.append(pos_l)
        # define edge_feats and edge_ref
        bs = pairs_coords.size(0)
        num_edges = pairs_coords.size(1)
        edge_positional_feat = list()
        coords = pairs_coords.long()
        coords = coords.view(bs, num_edges, 2, 2)
        for b_i in range(bs):
            positional_feat = pixel_features[b_i, coords[b_i, :, :, 1], coords[b_i, :, :, 0], :]
            edge_positional_feat.append(positional_feat)
        edge_positional_feat = torch.stack(edge_positional_feat, dim=0)
        edge_positional_feat = edge_positional_feat.view(bs, num_edges, -1)
        edge_feat = self.edge_input_fc(edge_positional_feat.view(bs * num_edges, -1))
        edge_feat = edge_feat.view(bs, num_edges, -1)
        edge_center = (pairs_coords[:, :, 0:2].float() + pairs_coords[:, :, 2:4].float()) / 2
        edge_center = edge_center / feat_mask.shape[1]
        # define grid_feats and grid_ref
        grid_feats = self.to_patch_embedding(pixel_features)
        H_tgt = W_tgt = int(np.sqrt(grid_feats.shape[1]))
        grid_ref = self.get_decoder_reference_points(H_tgt, W_tgt, grid_feats.device)
        query_embeds = self.query_embed.weight
        learnable_embed = self.learnable_embed.weight
        gifs, pred_coords, pred_logists = self.transformer(srcs, masks, all_pos, edge_feat, edge_center, grid_feats,
                                                           grid_ref, learnable_embed, query_embeds)
        return gifs, pred_coords, pred_logists


class ImplicitTransformer(nn.Module):
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
        # decoder_attn_layer = DeformableAttnDecoderLayer(d_model, dim_feedforward,
        #                                                 dropout, activation,
        #                                                 num_feature_levels, nhead, dec_n_points)
        # # one-layer decoder, without self-attention layers
        # self.per_edge_decoder = DeformableTransformerDecoder(decoder_attn_layer, 1, False, with_sa=False)
        # decoder for implicit prediction
        # inputs1: tgt, input2: reference points of tgt
        self.implicit_decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, False, with_sa=True)
        # decoder for vertex prediction
        vertex_decoder_layer = VertexDecoderLayer(d_model, dim_feedforward, dropout, activation,
                                                  num_feature_levels, nhead, dec_n_points)
        self.vertex_decoder = VertexDecoder(vertex_decoder_layer, num_decoder_layers, True, return_intermediate_dec,
                                            False, query_pos_type)
        # vertex_decoder_layer = VertexDecoderLayer(d_model, dim_feedforward, dropout, activation,
        #                                           num_feature_levels, nhead, dec_n_points)
        # self.implicit_decoder = VertexDecoder(vertex_decoder_layer, num_decoder_layers, True, return_intermediate_dec,
        #                                       False, query_pos_type)
        self.vertex_decoder.pos_trans = nn.Linear(d_model, d_model)
        self.vertex_decoder.pos_trans_norm = nn.LayerNorm(d_model)
        self.coords_embed = MLP(d_model, d_model, 2, 3)
        self.class_embed = nn.Linear(d_model, 1)
        self.class_embed = _get_clones(self.class_embed, self.vertex_decoder.num_layers)
        self.coords_embed = _get_clones(self.coords_embed, self.vertex_decoder.num_layers)
        self.vertex_decoder.coords_embed = self.coords_embed
        self.vertex_decoder.class_embed = self.class_embed
        # other properties
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.implicit_mlp = MLP(input_dim=d_model, hidden_dim=d_model // 2, output_dim=1, num_layers=2)
        # define self_attention Module and cross-attention module
        # 方案1： learnable embedding cross attention with memory and then self-attention
        # 方案2： learnable embedding uses deformable-attention with memory
        # define learnable queries
        self.decoder_cross_attn = PreNorm(d_model, Attention(d_model, d_model, heads=1, dim_head=d_model),
                                          context_dim=d_model)
        self.decoder_ff = PreNorm(d_model, FeedForward(d_model))
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

    # 直接inputs to the decoder也行其实
    # 试试直接用grid数据？
    def forward(self, srcs, masks, pos_embeds, edge_feats, edge_ref, grid_feats, grid_ref, learnable_embed, query_embeds):
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

        # edge_feats -- edge ref
        # grid_feats -- grid ref
        # define learnable embedding
        # define cross attention
        # edge_feats interact with learnable embedding
        bs, _, c = memory.shape

        # # HEAT way of decoding edge
        # interpolate_edge, _ = self.per_edge_decoder(edge_feats, edge_ref, memory, spatial_shapes, level_start_index,
        #                                             valid_ratios, edge_feats, mask_flatten)
        # f_gifs, _ = self.implicit_decoder(interpolate_edge, edge_ref, memory, spatial_shapes, level_start_index,
        #                                   valid_ratios, edge_feats, mask_flatten)
        # # input queries --- interpolate with image fusion --- cross attention with learnable features
        # gifs = self.implicit_mlp(f_gifs)
        # # HEAT way of decoding edge

        # learnable way of decoding queries
        # 1:cross-attention with memory doesn't work chachachacha
        # 2: cross-attention with implicit decoder latent
        learnable_embed = learnable_embed.unsqueeze(0).expand(bs, -1, -1)
        query_embeds = query_embeds.unsqueeze(0).expand(bs, -1, -1)
        reference_points = query_embeds.sigmoid()
        learnable_embed, _ = self.implicit_decoder(learnable_embed, reference_points, memory, spatial_shapes,
                                                   level_start_index,
                                                   valid_ratios, learnable_embed, mask_flatten)
        latents = self.decoder_cross_attn(edge_feats, context=learnable_embed)
        latents = latents + self.decoder_ff(latents)
        gifs = self.implicit_mlp(latents)
        # implicit decoder

        # learnable way of decoding queries

        # handle grid
        hs, inter_references, inter_classes = self.vertex_decoder(grid_feats, grid_ref, memory, src_flatten,
                                                                  spatial_shapes, level_start_index, valid_ratios)
        return gifs, inter_references, inter_classes


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
