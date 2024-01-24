import torch
import torch.nn as nn
import numpy as np
from models.mlp import MLP
from models.deformable_transformer import DeformableTransformerEncoderLayer, DeformableTransformerEncoder, \
    DeformableTransformerDecoder, DeformableTransformerDecoderLayer, DeformableAttnDecoderLayer
from models.ops.modules import MSDeformAttn
from models.corner_models import PositionEmbeddingSine
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
import torch.nn.functional as F
from utils.misc import NestedTensor
from einops.layers.torch import Rearrange
from models.resnet import convrelu
from models.resnet import ResNetBackbone


class GIFS_HEAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_features_levels, backbone_strides, backbone_num_channel):
        super(GIFS_HEAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_features_levels

        input_project_list = []  # project backbone feature maps
        for _ in range(len(backbone_strides)):
            # in_channels: backbone output channels
            in_channels = backbone_num_channel[_]
            input_project_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim)
            ))
        for _ in range(num_features_levels - len(backbone_strides)):
            input_project_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, hidden_dim)
            ))
            in_channels = hidden_dim
        self.input_project = nn.ModuleList(input_project_list)
        self.img_position_embedding = PositionEmbeddingSine(hidden_dim // 2)
        self.point_input_fc = nn.Linear(input_dim, hidden_dim)
        # define the point transformer
        self.transformer = PointTransformer(d_model=hidden_dim, nhead=8, num_encoder_layers=1,
                                            num_decoder_layers=6, dim_feedforward=1024, dropout=0.1)

    def get_ms_feat(self, xs, img_mask):
        out: Dict[str, NestedTensor] = {}
        for name, x in sorted(xs.items()):
            m = img_mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

    def forward(self, image_feats, feat_mask, image_positional_encoding, point_coords):
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

        batch_size = point_coords.size(0)
        points_num = point_coords.size(1)

        point_positional_feat = list()
        coords = point_coords.long()
        for b_i in range(batch_size):
            positional_feat = image_positional_encoding[b_i, coords[b_i, :, 0], coords[b_i, :, 1], :]
            point_positional_feat.append(positional_feat)
        point_positional_feat = torch.stack(point_positional_feat, dim=0)
        point_positional_feat = point_positional_feat.view(batch_size, points_num, -1)

        point_feat = self.point_input_fc(point_positional_feat.view(batch_size * points_num, -1))
        point_feat = point_feat.view(batch_size, points_num, -1)

        point_coords = point_coords / feat_mask.shape[1]
        output = self.transformer(input_features, masks, features_pos, point_feat, point_coords)
        return output


class PointTransformer(nn.Module):
    def __init__(
            self, d_model=512, nhead=8, num_encoder_layers=6,
            num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
            activation="relu", return_intermediate_dec=False,
            num_feature_levels=4, dec_n_points=4, enc_n_points=4
    ):
        super(PointTransformer, self).__init__()
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_attn_layer = DeformableAttnDecoderLayer(d_model, dim_feedforward,
                                                        dropout, activation,
                                                        num_feature_levels, nhead, dec_n_points)
        # one-layer decoder, without self-attention layers
        self.per_edge_decoder = DeformableTransformerDecoder(decoder_attn_layer, 1, False, with_sa=False)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        # edge decoder w/ self-attention layers (image-aware decoder and geom-only decoder)
        self.relational_decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers,
                                                               return_intermediate_dec, with_sa=True)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.output_fc = MLP(input_dim=d_model, hidden_dim=d_model // 2, output_dim=1, num_layers=2)
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

    def forward(self, features, masks, pos_features, point_feats, reference_points):
        # flatten the corresponding matrix
        features_flatten = []
        spatial_shapes = []
        feature_pos_flatten = []
        mask_flatten = []
        for level, (feature, mask, pos_feature) in enumerate(zip(features, masks, pos_features)):
            bs, c, h, w = feature.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feature = feature.flatten(2).transpose(1, 2)
            features_flatten.append(feature)
            mask = mask.flatten(1)
            pos_feature = pos_feature.flatten(2).transpose(1, 2)
            pos_feature = pos_feature + self.level_embed[level].view(1, 1, -1)
            feature_pos_flatten.append(pos_feature)
            mask_flatten.append(mask)

        features_flatten = torch.cat(features_flatten, 1)
        feature_pos_flatten = torch.cat(feature_pos_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=features_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # input to the encoder
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        transformer_features = self.encoder(features_flatten, spatial_shapes, level_start_index, valid_ratios,
                                            feature_pos_flatten, mask_flatten)
        bs, _, c = transformer_features.shape

        tgt = point_feats
        f, _ = self.per_edge_decoder(tgt, reference_points, transformer_features, spatial_shapes, level_start_index,
                                     valid_ratios, tgt, mask_flatten)
        f_coord = point_feats
        # decoder
        hs, inter_references = self.relational_decoder(f, reference_points, transformer_features, spatial_shapes,
                                                       level_start_index, valid_ratios, f_coord, mask_flatten)
        output = self.output_fc(hs)
        return output


class GIFS_HEATv1(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GIFS_HEATv1, self).__init__()
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
        # define the point transformer
        # self.transformer = EdgeTransformer(d_model=hidden_dim, nhead=8, num_encoder_layers=1,
        #                                    num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
        #                                    num_feature_levels=num_features_levels, dec_n_points=num_features_levels,
        #                                    enc_n_points=num_features_levels)
        self.transformer = EdgeTransformer(d_model=hidden_dim, nhead=8, num_encoder_layers=6,
                                           num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                                           num_feature_levels=4, dec_n_points=4,
                                           enc_n_points=4)

    def get_ms_feat(self, xs, img_mask):
        out: Dict[str, NestedTensor] = {}
        for name, x in sorted(xs.items()):
            m = img_mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

    def forward(self, inputs, image_positional_encoding, edge_coords):
        image_feats, feat_mask, _ = self.backbone(inputs)
        # pixels, pixel_features = get_pixel_features(image_size=256)
        # pixel_features = pixel_features.cuda()
        # pixel_features = pixel_features.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
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

        # edge positional encoding
        edge_positional_feat = list()
        coords = edge_coords.long()
        coords = coords.view(batch_size, edges_num, 2, 2)
        for b_i in range(batch_size):
            positional_feat = image_positional_encoding[b_i, coords[b_i, :, :, 1], coords[b_i, :, :, 0], :]
            edge_positional_feat.append(positional_feat)
        edge_positional_feat = torch.stack(edge_positional_feat, dim=0)
        edge_positional_feat = edge_positional_feat.view(batch_size, edges_num, -1)

        # positional encoding -- feat

        edge_feat = self.edge_input_fc(edge_positional_feat.view(batch_size * edges_num, -1))
        edge_feat = edge_feat.view(batch_size, edges_num, -1)
        edge_center = (edge_coords[:, :, 0:2].float() + edge_coords[:, :, 2:4].float()) / 2
        edge_center = edge_center / feat_mask.shape[1]

        gifs_hb = self.transformer(input_features, masks, features_pos, edge_feat, edge_center)
        return gifs_hb


class EdgeTransformer(nn.Module):
    def __init__(
            self, d_model=512, nhead=8, num_encoder_layers=6,
            num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
            activation="relu", return_intermediate_dec=False,
            num_feature_levels=4, dec_n_points=4, enc_n_points=4
    ):
        super(EdgeTransformer, self).__init__()
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_attn_layer = DeformableAttnDecoderLayer(d_model, dim_feedforward,
                                                        dropout, activation,
                                                        num_feature_levels, nhead, dec_n_points)
        # one-layer decoder, without self-attention layers
        self.per_edge_decoder = DeformableTransformerDecoder(decoder_attn_layer, 1, False, with_sa=False)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        # edge decoder w/ self-attention layers (image-aware decoder and geom-only decoder)

        self.gt_label_embed = nn.Embedding(3, d_model)
        self.gifs_decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers,
                                                         return_intermediate_dec, with_sa=True)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.hb_gifs_fc = MLP(input_dim=d_model, hidden_dim=d_model // 2, output_dim=1, num_layers=2)
        self.rel_gifs_fc = MLP(input_dim=d_model, hidden_dim=d_model // 2, output_dim=2, num_layers=2)

        self.input_fc_hb = MLP(input_dim=2 * d_model, hidden_dim=d_model, output_dim=d_model, num_layers=2)
        self.input_fc_rel = MLP(input_dim=2 * d_model, hidden_dim=d_model, output_dim=d_model, num_layers=2)
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

    def candidate_filtering(self, conf_per_edge, f_img_edge, edge_feats, edge_reference_points):
        B, L, _ = f_img_edge.shape
        preds = conf_per_edge.detach().softmax(1)[:, :, 0]
        sorted_ids = torch.argsort(preds, dim=-1, descending=True)
        filtered_hb = list()
        filtered_query = list()
        filtered_rp = list()
        selected_ids = list()
        for b_i in range(B):
            ids = sorted_ids[b_i, :1000]
            filtered_hb.append(f_img_edge[b_i][ids])
            filtered_query.append(edge_feats[b_i][ids])
            filtered_rp.append(edge_reference_points[b_i][ids])
            selected_ids.append(ids)
        filtered_hb = torch.stack(filtered_hb, dim=0)
        filtered_query = torch.stack(filtered_query, dim=0)
        filtered_rp = torch.stack(filtered_rp, dim=0)
        selected_ids = torch.stack(selected_ids, dim=0)
        return filtered_hb, filtered_query, filtered_rp, selected_ids

    def forward(self, features, masks, pos_features, edge_feats, edge_reference_points):
        # flatten the corresponding matrix
        features_flatten = []
        spatial_shapes = []
        feature_pos_flatten = []
        mask_flatten = []
        for level, (feature, mask, pos_feature) in enumerate(zip(features, masks, pos_features)):
            bs, c, h, w = feature.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feature = feature.flatten(2).transpose(1, 2)
            features_flatten.append(feature)
            mask = mask.flatten(1)
            pos_feature = pos_feature.flatten(2).transpose(1, 2)
            pos_feature = pos_feature + self.level_embed[level].view(1, 1, -1)
            feature_pos_flatten.append(pos_feature)
            mask_flatten.append(mask)

        features_flatten = torch.cat(features_flatten, 1)
        feature_pos_flatten = torch.cat(feature_pos_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=features_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # input to the encoder
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        transformer_features = self.encoder(features_flatten, spatial_shapes, level_start_index, valid_ratios,
                                            feature_pos_flatten, mask_flatten)
        bs, _, c = transformer_features.shape

        f_img_edge, _ = self.per_edge_decoder(edge_feats, edge_reference_points, transformer_features, spatial_shapes,
                                              level_start_index,
                                              valid_ratios, edge_feats, mask_flatten)
        # conf_per_edge = self.conf_edge_fc(f_img_edge)
        #
        # filtered_hb, filtered_query, filtered_rp, selected_ids = self.candidate_filtering(
        #     conf_per_edge, f_img_edge, edge_feats, edge_reference_points,
        # )

        hb_gifs, inter_references = self.gifs_decoder(f_img_edge, edge_reference_points, transformer_features,
                                                      spatial_shapes,
                                                      level_start_index, valid_ratios, edge_feats, mask_flatten)
        gifs_hb = self.hb_gifs_fc(hb_gifs)

        # rel_gifs, _ = self.gifs_decoder(edge_feats, edge_reference_points, transformer_features,
        #                                 spatial_shapes,
        #                                 level_start_index, valid_ratios, edge_feats, mask_flatten)
        # gifs_rel = self.rel_gifs_fc(rel_gifs)

        return gifs_hb


class GIFS_HEATv2(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GIFS_HEATv2, self).__init__()
        self.backbone = ResNetBackbone()
        self.num_features_levels = 4
        backbone_strides = self.backbone.strides
        backbone_num_channel = self.backbone.num_channels

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # self.num_feature_levels = num_features_levels

        input_project_list = []  # project backbone feature maps
        for _ in range(len(backbone_strides)):
            # in_channels: backbone output channels
            in_channels = backbone_num_channel[_]
            input_project_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim)
            ))
        for _ in range(self.num_features_levels - len(backbone_strides)):
            input_project_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, hidden_dim)
            ))
            in_channels = hidden_dim
        self.input_project = nn.ModuleList(input_project_list)
        self.patch_size = 4
        patch_dim = (self.patch_size ** 2) * input_dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, input_dim),
            nn.Linear(input_dim, hidden_dim),
        )
        self.point_input_fc = nn.Linear(input_dim, hidden_dim)
        self.img_position_embedding = PositionEmbeddingSine(hidden_dim // 2)
        # define the point transformer
        self.transformer = CornerTransformer(d_model=hidden_dim, nhead=8, num_encoder_layers=6,
                                             dim_feedforward=1024, dropout=0.1)

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
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                                      torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / height
        ref_x = ref_x.reshape(-1)[None] / width
        ref = torch.stack((ref_x, ref_y), -1)
        return ref

    def forward(self, inputs, pixel_feat, pixel):
        image_feats, feat_mask, all_image_feats = self.backbone(inputs)
        # pixels, pixel_features = get_pixel_features(image_size=256)
        # pixel_features = pixel_features.cuda()
        # pixel_features = pixel_features.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
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

        if self.num_features_levels > len(input_features):
            for level in range(len(input_features), self.num_features_levels):
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

        sp_inputs = self.to_patch_embedding(pixel_feat)

        H_tgt = W_tgt = int(np.sqrt(sp_inputs.shape[1]))
        reference_points_s1 = self.get_decoder_reference_points(H_tgt, W_tgt, sp_inputs.device)

        output = self.transformer(input_features, masks, features_pos, sp_inputs, reference_points_s1, all_image_feats)
        return output


class CornerTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4,
                 ):
        super(CornerTransformer, self).__init__()

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_attn_layer = DeformableAttnDecoderLayer(d_model, dim_feedforward,
                                                        dropout, activation,
                                                        num_feature_levels, nhead, dec_n_points)
        self.per_edge_decoder = DeformableTransformerDecoder(decoder_attn_layer, 1, False, with_sa=False)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        # upconv layers
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = convrelu(256 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, d_model, 3, 1)
        self.output_fc_1 = nn.Linear(d_model, 1)
        self.output_fc_2 = nn.Linear(d_model, 1)

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

    def forward(self, srcs, masks, pos_embeds, query_embed, reference_points, all_image_feats):
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

        tgt = query_embed

        # relational decoder
        hs_pixels_s1, _ = self.per_edge_decoder(tgt, reference_points, memory,
                                                spatial_shapes, level_start_index, valid_ratios, query_embed,
                                                mask_flatten)

        feats_s1, preds_s1 = self.generate_corner_preds(hs_pixels_s1, all_image_feats)

        return preds_s1

    def generate_corner_preds(self, outputs, conv_outputs):
        B, L, C = outputs.shape
        side = int(np.sqrt(L))
        outputs = outputs.view(B, side, side, C)
        outputs = outputs.permute(0, 3, 1, 2)
        outputs = torch.cat([outputs, conv_outputs['layer1']], dim=1)
        x = self.conv_up1(outputs)

        x = self.upsample(x)
        x = torch.cat([x, conv_outputs['layer0']], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, conv_outputs['x_original']], dim=1)
        x = self.conv_original_size2(x)

        logits = x.permute(0, 2, 3, 1)
        preds = self.output_fc_1(logits)
        preds = preds.squeeze(-1).sigmoid()
        return logits, preds


class GIFS_HEATv3(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_features_levels, backbone_strides, backbone_num_channel):
        super(GIFS_HEATv3, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_features_levels

        input_project_list = []  # project backbone feature maps
        for _ in range(len(backbone_strides)):
            # in_channels: backbone output channels
            in_channels = backbone_num_channel[_]
            input_project_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim)
            ))
        for _ in range(num_features_levels - len(backbone_strides)):
            input_project_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, hidden_dim)
            ))
            in_channels = hidden_dim
        self.input_project = nn.ModuleList(input_project_list)
        self.img_position_embedding = PositionEmbeddingSine(hidden_dim // 2)
        self.edge_input_fc = nn.Linear(input_dim * 2, hidden_dim)
        self.patch_size = 4
        patch_dim = (self.patch_size ** 2) * input_dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, input_dim),
            nn.Linear(input_dim, hidden_dim),
        )

        self.transformer = GIFSTransformer(d_model=hidden_dim, nhead=8, num_encoder_layers=1,
                                           num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                                           num_feature_levels=num_features_levels, dec_n_points=num_features_levels,
                                           enc_n_points=num_features_levels)

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
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                                      torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / height
        ref_x = ref_x.reshape(-1)[None] / width
        ref = torch.stack((ref_x, ref_y), -1)
        return ref

    def forward(self, image_feats, feat_mask, image_positional_encoding, edge_coords, all_image_feats):
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
                mask = F.interpolate(m[None].float(), size=proj_feat.shape[-2:]).to(torch.bool)[0].to(
                    proj_feat.device)
                masks.append(mask)
                input_features.append(proj_feat)
                feat_pos = self.img_position_embedding(proj_feat).to(proj_feat.dtype)
                features_pos.append(feat_pos)

        batch_size = edge_coords.size(0)
        edges_num = edge_coords.size(1)

        # edge positional encoding
        edge_positional_feat = list()
        coords = edge_coords.long()
        coords = coords.view(batch_size, edges_num, 2, 2)
        for b_i in range(batch_size):
            positional_feat = image_positional_encoding[b_i, coords[b_i, :, :, 1], coords[b_i, :, :, 0], :]
            edge_positional_feat.append(positional_feat)
        edge_positional_feat = torch.stack(edge_positional_feat, dim=0)
        edge_positional_feat = edge_positional_feat.view(batch_size, edges_num, -1)

        # positional encoding -- feat

        edge_feat = self.edge_input_fc(edge_positional_feat.view(batch_size * edges_num, -1))
        edge_feat = edge_feat.view(batch_size, edges_num, -1)
        edge_center = (edge_coords[:, :, 0:2].float() + edge_coords[:, :, 2:4].float()) / 2
        edge_center = edge_center / feat_mask.shape[1]

        sp_inputs = self.to_patch_embedding(image_positional_encoding)
        H_tgt = W_tgt = int(np.sqrt(sp_inputs.shape[1]))
        reference_points_s1 = self.get_decoder_reference_points(H_tgt, W_tgt, sp_inputs.device)

        gifs, point_pred = self.transformer(input_features, masks, features_pos, edge_feat, edge_center, sp_inputs,
                                            reference_points_s1, all_image_feats)
        return gifs, point_pred


class GIFSTransformer(nn.Module):
    def __init__(
            self, d_model=512, nhead=8, num_encoder_layers=6,
            num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
            activation="relu", return_intermediate_dec=False,
            num_feature_levels=4, dec_n_points=4, enc_n_points=4
    ):
        super(GIFSTransformer, self).__init__()
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_attn_layer = DeformableAttnDecoderLayer(d_model, dim_feedforward,
                                                        dropout, activation,
                                                        num_feature_levels, nhead, dec_n_points)
        # one-layer decoder, without self-attention layers
        self.img_feature_fusion_decoder = DeformableTransformerDecoder(decoder_attn_layer, 1, False, with_sa=False)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        # edge decoder w/ self-attention layers (image-aware decoder and geom-only decoder)
        self.gifs_decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers,
                                                         return_intermediate_dec, with_sa=True)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.gifs_fc = MLP(input_dim=d_model, hidden_dim=d_model // 2, output_dim=1, num_layers=2)

        # upconv layers
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = convrelu(256 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, d_model, 3, 1)
        self.output_fc_1 = nn.Linear(d_model, 1)
        self.output_fc_2 = nn.Linear(d_model, 1)

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

    def forward(self, features, masks, pos_features, edge_feats, edge_reference_points, sp_inputs, reference_points_s1,
                all_image_feats):
        # flatten the corresponding matrix
        features_flatten = []
        spatial_shapes = []
        feature_pos_flatten = []
        mask_flatten = []
        for level, (feature, mask, pos_feature) in enumerate(zip(features, masks, pos_features)):
            bs, c, h, w = feature.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feature = feature.flatten(2).transpose(1, 2)
            features_flatten.append(feature)
            mask = mask.flatten(1)
            pos_feature = pos_feature.flatten(2).transpose(1, 2)
            pos_feature = pos_feature + self.level_embed[level].view(1, 1, -1)
            feature_pos_flatten.append(pos_feature)
            mask_flatten.append(mask)

        features_flatten = torch.cat(features_flatten, 1)
        feature_pos_flatten = torch.cat(feature_pos_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=features_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # input to the encoder
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        transformer_features = self.encoder(features_flatten, spatial_shapes, level_start_index, valid_ratios,
                                            feature_pos_flatten, mask_flatten)
        bs, _, c = transformer_features.shape

        tgt = sp_inputs

        # relational decoder
        hs_pixels_s1, _ = self.img_feature_fusion_decoder(tgt, reference_points_s1, transformer_features,
                                                          spatial_shapes, level_start_index, valid_ratios, sp_inputs,
                                                          mask_flatten)

        f_img_edge, _ = self.img_feature_fusion_decoder(edge_feats, edge_reference_points, transformer_features,
                                                        spatial_shapes,
                                                        level_start_index,
                                                        valid_ratios, edge_feats, mask_flatten)

        hs_gifs, inter_references = self.gifs_decoder(f_img_edge, edge_reference_points, transformer_features,
                                                      spatial_shapes,
                                                      level_start_index, valid_ratios, edge_feats, mask_flatten)
        gifs = self.gifs_fc(hs_gifs)
        feats_s1, point_pred = self.generate_corner_preds(hs_pixels_s1, all_image_feats)

        return gifs, point_pred

    def generate_corner_preds(self, outputs, conv_outputs):
        B, L, C = outputs.shape
        side = int(np.sqrt(L))
        outputs = outputs.view(B, side, side, C)
        outputs = outputs.permute(0, 3, 1, 2)
        outputs = torch.cat([outputs, conv_outputs['layer1']], dim=1)
        x = self.conv_up1(outputs)

        x = self.upsample(x)
        x = torch.cat([x, conv_outputs['layer0']], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, conv_outputs['x_original']], dim=1)
        x = self.conv_original_size2(x)

        logits = x.permute(0, 2, 3, 1)
        preds = self.output_fc_1(logits)
        preds = preds.squeeze(-1).sigmoid()
        return logits, preds
