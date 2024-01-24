# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
import copy
from models.ops.modules import MSDeformAttn
from utils.misc import inverse_sigmoid, NestedTensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from models.resnet import ResNetBackbone
from einops.layers.torch import Rearrange
import numpy as np


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", poly_refine=True, return_intermediate_dec=False, aux_loss=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4, query_pos_type="none"):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, poly_refine,
                                                    return_intermediate_dec, aux_loss, query_pos_type)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if query_pos_type == 'sine':
            self.decoder.pos_trans = nn.Linear(d_model, d_model)
            self.decoder.pos_trans_norm = nn.LayerNorm(d_model)

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
        assert query_embed is not None

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

        query_embed = query_embed.expand(bs, -1, -1)
        # tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = query_embed
        init_reference_out = reference_points

        # decoder
        hs, inter_references, inter_classes = self.decoder(tgt, reference_points, memory, src_flatten,
                                                           spatial_shapes, level_start_index, valid_ratios, query_embed,
                                                           mask_flatten, tgt_masks)
        # add another decoder to predict binary flags
        return hs, init_reference_out, inter_references, inter_classes


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
                              padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                src_padding_mask=None, tgt_masks=None):
        # self attention
        # tgt: D^i, query_pos: P^i, reference_points_input: ref
        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), attn_mask=tgt_masks)[
            0].transpose(0, 1)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, poly_refine=True, return_intermediate=False, aux_loss=False,
                 query_pos_type='none'):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.poly_refine = poly_refine
        self.return_intermediate = return_intermediate
        self.aux_loss = aux_loss
        self.query_pos_type = query_pos_type

        self.coords_embed = None
        self.class_embed = None
        self.pos_trans = None
        self.pos_trans_norm = None

    def get_query_pos_embed(self, ref_points):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=ref_points.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)  # [128]
        # N, L, 2
        ref_points = ref_points * scale
        # N, L, 2, 128
        pos = ref_points[:, :, :, None] / dim_t
        # N, L, 256
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def forward(self, tgt, reference_points, src, src_flatten, src_spatial_shapes, src_level_start_index,
                src_valid_ratios,
                query_pos=None, src_padding_mask=None, tgt_masks=None):
        output = tgt  # [10, 800, 256]

        intermediate = []
        intermediate_reference_points = []
        intermediate_classes = []
        point_classes = torch.zeros(output.shape[:2]).unsqueeze(-1).to(output.device)
        for lid, layer in enumerate(self.layers):

            assert reference_points.shape[-1] == 2
            reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            # reference point -- pos_embed -- MLP -- norm -- query pos -- P^i
            if self.query_pos_type == 'sine':
                query_pos = self.pos_trans_norm(self.pos_trans(self.get_query_pos_embed(reference_points)))

            elif self.query_pos_type == 'none':
                query_pos = None
            # output: D^i, query_pos: P^i, reference_points_input: ref
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index,
                           src_padding_mask, tgt_masks)

            # iterative polygon refinement
            if self.poly_refine:
                offset = self.coords_embed[lid](output)
                assert reference_points.shape[-1] == 2
                new_reference_points = offset
                new_reference_points = offset + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points
                # reference_points = reference_points + offset

            # if not using iterative polygon refinement, just output the reference points decoded from the last layer
            elif lid == len(self.layers) - 1:
                offset = self.coords_embed[-1](output)
                assert reference_points.shape[-1] == 2
                new_reference_points = offset
                new_reference_points = offset + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points
                # reference_points = reference_points + offset

            # If aux loss supervision, we predict classes label from each layer and supervise loss
            if self.aux_loss:
                point_classes = self.class_embed[lid](output)
            # Otherwise, we only predict class label from the last layer
            elif lid == len(self.layers) - 1:
                point_classes = self.class_embed[-1](output)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_classes.append(point_classes)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), torch.stack(
                intermediate_classes)

        return output, reference_points, point_classes


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


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        mask = torch.zeros([x.shape[0], x.shape[2], x.shape[3]]).bool().to(x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class VertexModel(nn.Module):

    def __init__(self, input_dim, args):
        super(VertexModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim

        self.backbone = ResNetBackbone()
        backbone_strides = self.backbone.strides
        backbone_num_channel = self.backbone.num_channels
        self.transformer = DeformableTransformer(
            d_model=args.hidden_dim,
            nhead=args.nheads,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            activation="relu",
            poly_refine=args.with_poly_refine,
            return_intermediate_dec=True,
            aux_loss=args.aux_loss,
            num_feature_levels=args.num_feature_levels,
            dec_n_points=args.dec_n_points,
            enc_n_points=args.enc_n_points,
            query_pos_type=args.query_pos_type)

        self.num_feature_levels = 4

        input_project_list = []  # project backbone feature maps
        for _ in range(len(backbone_strides)):
            # in_channels: backbone output channels
            in_channels = backbone_num_channel[_]
            input_project_list.append(nn.Sequential(
                nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                nn.GroupNorm(32, self.hidden_dim)
            ))
        for _ in range(self.num_feature_levels - len(backbone_strides)):
            input_project_list.append(nn.Sequential(
                nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, self.hidden_dim)
            ))
            in_channels = self.hidden_dim
        self.input_project = nn.ModuleList(input_project_list)
        self.patch_size = 8
        patch_dim = (self.patch_size ** 2) * self.input_dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
        )
        self.img_position_embedding = PositionEmbeddingSine(self.hidden_dim // 2)
        self.coords_embed = MLP(self.hidden_dim, self.hidden_dim, 2, 3)
        self.class_embed = nn.Linear(self.hidden_dim, args.num_classes)
        self.with_poly_refine = args.with_poly_refine
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(args.num_classes) * bias_value
        nn.init.constant_(self.coords_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.coords_embed.layers[-1].bias.data, 0)

        for proj in self.input_project:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        num_pred = self.transformer.decoder.num_layers
        if self.with_poly_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.coords_embed = _get_clones(self.coords_embed, num_pred)
            nn.init.constant_(self.coords_embed[0].layers[-1].bias.data[2:], -2.0)
        self.transformer.decoder.coords_embed = self.coords_embed
        self.transformer.decoder.class_embed = self.class_embed

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

    def forward(self, inputs, pixel_feats):
        # process image features
        # query_embeds: the center coordinate of the vertex
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
            srcs.append(self.input_project[l](src))
            pos = self.img_position_embedding(src).to(src.dtype)
            all_pos.append(pos)
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_project[l](features[-1].tensors)
                else:
                    src = self.input_project[l](srcs[-1])
                m = feat_mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0].to(src.device)
                pos_l = self.img_position_embedding(src).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                all_pos.append(pos_l)

        sp_inputs = self.to_patch_embedding(pixel_feats)

        # compute the reference points
        H_tgt = W_tgt = int(np.sqrt(sp_inputs.shape[1]))
        reference_points_s1 = self.get_decoder_reference_points(H_tgt, W_tgt, sp_inputs.device)
        tgt_embeds = sp_inputs
        hs, init_reference, inter_references, inter_classes = self.transformer(srcs, masks, all_pos, reference_points_s1, tgt_embeds)

        return inter_classes[-1], inter_references[-1]
