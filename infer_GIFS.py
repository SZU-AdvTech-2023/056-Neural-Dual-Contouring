import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import floor_data
from datasets.outdoor_buildings import OutdoorBuildingDataset
from datasets.s3d_floorplans import S3DFloorplanDataset
from datasets.data_utils import collate_fn, get_pixel_features
from models.resnet import ResNetBackbone
from models.GIFS_HEAT import GIFS_HEAT, GIFS_HEATv1, GIFS_HEATv2
from models.GIFS_Former import GIFS_Former
import numpy as np
import cv2
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
from metrics.get_metric import compute_metrics, get_recall_and_precision
import skimage
from scipy.ndimage import gaussian_filter
import argparse
from floor_data import FloorPlanDataset
from models.VertexFormer import VertexModel
from metrics.new_utils import *

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

datasets_folder = 'data/s3d_floorplan'

def get_args_parser():
    parser = argparse.ArgumentParser('Vertex', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--lr_drop', default=[400], type=list)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--num_queries', default=800, type=int,
                        help="Number of query slots (num_polys * max. number of corner per poly)")
    parser.add_argument('--num_polys', default=20, type=int,
                        help="Number of maximum number of room polygons")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--query_pos_type', default='sine', type=str, choices=('static', 'sine', 'none'),
                        help="Type of query pos in decoder - \
                        1. static: same setting with DETR and Deformable-DETR, the query_pos is the same for all layers \
                        2. sine: since embedding from reference points (so if references points update, query_pos also \
                        3. none: remove query_pos")
    parser.add_argument('--with_poly_refine', default=True, action='store_true',
                        help="iteratively refine reference points (i.e. positional part of polygon queries)")
    parser.add_argument('--masked_attn', default=False, action='store_true',
                        help="if true, the query in one room will not be allowed to attend other room")
    parser.add_argument('--semantic_classes', default=-1, type=int,
                        help="Number of classes for semantically-rich floorplan:  \
                        1. default -1 means non-semantic floorplan \
                        2. 19 for Structured3D: 16 room types + 1 door + 1 window + 1 empty")

    # loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_true',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_coords', default=5, type=float,
                        help="L1 coords coefficient in the matching cost")

    # loss coefficients
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--room_cls_loss_coef', default=0.2, type=float)
    parser.add_argument('--coords_loss_coef', default=5, type=float)
    parser.add_argument('--raster_loss_coef', default=1, type=float)

    # dataset parameters
    parser.add_argument('--dataset_name', default='stru3d')
    parser.add_argument('--dataset_root', default='data/stru3d', type=str)

    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--job_name', default='train_stru3d', type=str)

    return parser

def read_list(filename):
    data_list = []
    file_path = os.path.join(datasets_folder, filename + '.txt')
    with open(file_path, 'r', encoding='utf-8') as infile:
        for name in infile:
            data_name = name.strip('\n').split()[0]
            data_list.append(data_name)
    return data_list


class Infer():
    def __init__(self):
        self.backbone = ResNetBackbone()
        self.strides = self.backbone.strides
        self.num_channels = self.backbone.num_channels
        # self.point_model = GIFS_HEAT(input_dim=128, hidden_dim=256, num_features_levels=4,
        #                              backbone_strides=self.strides,
        #                              backbone_num_channel=self.num_channels)
        self.edge_model = GIFS_HEATv1(
            input_dim=128, hidden_dim=256, num_features_levels=4,
            backbone_strides=self.strides,
            backbone_num_channel=self.num_channels
        )
        self.gpu = 0
        self.device = self.device = torch.device("cuda:" + str(self.gpu))
        self.dataset = floor_data.FloorPlanDataset(
            'train',
            data_dir='data/20230307_ndc',

        )
        self.data_loader = DataLoader(self.dataset, batch_size=1)

    def infer(self):
        self.backbone = self.backbone.to(self.device)
        self.edge_model = self.edge_model.to(self.device)
        for batch in self.data_loader:
            p = batch.get('grid_coords').to(self.device)
            inputs = batch.get('inputs').to(self.device)
            image_feats, feat_mask, _ = self.backbone(inputs)
            pixels, pixel_features = get_pixel_features(image_size=256)
            pixel_features = pixel_features.to(self.device)
            pixel_features = pixel_features.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
            pred = self.edge_model(image_feats, feat_mask, pixel_features, p)
            print(pred)


def load_edge_model(model_path):
    # backbone = ResNetBackbone()
    # strides = backbone.strides
    # num_channels = backbone.num_channels
    # backbone = nn.DataParallel(backbone)
    # backbone = backbone.cuda()
    # backbone.eval()
    # edge_model = GIFS_HEATv1(input_dim=128, hidden_dim=256, num_features_levels=4, backbone_strides=strides,
    #                          backbone_num_channel=num_channels)
    # edge_model = nn.DataParallel(edge_model)
    # edge_model = edge_model.cuda()
    # edge_model.eval()
    # ckpt = torch.load(model_path)
    # backbone.load_state_dict(ckpt['backbone_state_dict'])
    # edge_model.load_state_dict(ckpt['point_model_state_dict'])
    edge_model = GIFS_HEATv1(input_dim=128, hidden_dim=256)
    edge_model = edge_model.cuda()
    edge_model.eval()
    # edge_model = DDP(edge_model, device_ids=[0], find_unused_parameters=True)
    ckpt = torch.load(model_path)
    state_dict = {
        k.replace("module.", ""): ckpt["edge_model_state_dict"][k] for k in ckpt["edge_model_state_dict"]
    }
    # self.model.load_state_dict(state_dict)
    edge_model.load_state_dict(state_dict)
    # return backbone, edge_model
    return edge_model


def load_point_model(model_path, args):
    # backbone = ResNetBackbone()
    # strides = backbone.strides
    # num_channels = backbone.num_channels
    # backbone = nn.DataParallel(backbone)
    # backbone = backbone.cuda()
    # backbone.eval()
    # corner_model = GIFS_HEATv2(input_dim=128, hidden_dim=256, num_features_levels=4, backbone_strides=strides,
    #                            backbone_num_channel=num_channels)
    # corner_model = nn.DataParallel(corner_model)
    # corner_model = corner_model.cuda()
    # corner_model.eval()
    # ckpt = torch.load(model_path)
    # backbone.load_state_dict(ckpt['backbone_state_dict'])
    # corner_model.load_state_dict(ckpt['point_model_state_dict'])
    #
    # return backbone, corner_model
    model = VertexModel(input_dim=128, args=args)
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['model_state_dict'])
    return model



def load_img(img_name, is_raw=True):
    merge_folder = 'data/s3d_floorplan/merge'
    img_path = os.path.join(datasets_folder, 'density', img_name + '.png')
    density_path = os.path.join(datasets_folder, 'density', img_name + '.png')
    normal_path = os.path.join(datasets_folder, 'normals', img_name + '.png')
    density = cv2.imread(density_path)
    normal = cv2.imread(normal_path)
    if is_raw:
        # img = np.maximum(density, normal)
        img = density
        img = skimage.img_as_float(img)
        img = img.transpose((2, 0, 1))
        img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[:, np.newaxis, np.newaxis]
        img = img.astype(np.float32)
        img = torch.tensor(img, dtype=torch.float32).cuda().unsqueeze(0)
        vis_img = density
    else:
        img = cv2.imread(os.path.join('data/gt_img', img_name + '.png'))
        img = skimage.img_as_float(img)
        img = img.transpose((2, 0, 1))
        img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[:, np.newaxis, np.newaxis]
        img = img.astype(np.float32)
        img = torch.tensor(img, dtype=torch.float32).cuda().unsqueeze(0)
        vis_img = cv2.imread(os.path.join('data/gt_img', img_name + '.png'))
    return img, vis_img


def get_corner_labels(corners):
    image_size = 256
    labels = np.zeros((image_size, image_size))
    corners = corners.round()
    xint, yint = corners[:, 0].astype(np.int64), corners[:, 1].astype(np.int64)
    labels[xint, yint] = 1
    gauss_labels = gaussian_filter(labels, sigma=2)
    gauss_labels = gauss_labels / gauss_labels.max()
    return labels, gauss_labels

def vertex2mask(vertex, grid_size=32):
    mask = np.zeros([256, 256], np.float32)
    size_grid = int(256 / grid_size)
    vertex_idx = np.array(vertex / size_grid, dtype=np.int64)
    mask_offset = size_grid
    for idx in vertex_idx:
        x1 = max(0, idx[1] * size_grid - mask_offset)
        x2 = min(255, (idx[1] + 1) * size_grid + mask_offset)
        y1 = max(0, idx[0] * size_grid - mask_offset)
        y2 = min(255, (idx[0] + 1) * size_grid + mask_offset)
        mask[x1: x2, y1: y2] = 1
    return mask

def convert_annot(annot):
    corners = np.array(list(annot.keys()))
    corners_mapping = {tuple(c): idx for idx, c in enumerate(corners)}
    edges = set()
    for corner, connections in annot.items():
        idx_c = corners_mapping[tuple(corner)]
        for other_c in connections:
            idx_other_c = corners_mapping[tuple(other_c)]
            if (idx_c, idx_other_c) not in edges and (idx_other_c, idx_c) not in edges:
                edges.add((idx_c, idx_other_c))
    edges = np.array(list(edges))
    gt_data = {
        'corners': corners,
        'edges': edges
    }
    return gt_data


def load_label(label_dir, img_name):
    label_name = os.path.join(label_dir, img_name + '.npz')
    label_dict = np.load(label_name, allow_pickle=True)
    coords = label_dict['coords']
    flags = label_dict['flags']
    vertex_coords = label_dict['vertex_coords']
    vertex_flags = label_dict['vertex_flags']
    coords[coords == 256.0] = 255.0
    # new_coords = []
    # for coord in coords:
    #     dist = np.sqrt(np.sum((coord[0:2] - coord[2:4]) ** 2))
    #     if dist <= 8:
    #         new_coords.append(coord)
    # new_coords = np.array(new_coords, dtype=np.float32)
    # coords = new_coords
    # grid_size = 32
    # size_grid = int(256 / grid_size)
    select_idx = np.random.choice(len(flags), 1000, replace=False)
    coords = coords[select_idx, :]
    flags = flags[select_idx, :]
    annot_folder = 'data/s3d_floorplan/annot'
    annot_path = os.path.join(annot_folder, img_name + '.npy')
    annot = np.load(annot_path, allow_pickle=True, encoding='latin1').tolist()
    gt_data = convert_annot(annot)
    return {
        'coords': coords,
        'flags': flags,
        'vertex_coords': vertex_coords,
        'vertex_flags': vertex_flags,
        "gt_data": gt_data
    }


def load_data(img_name):
    label_dir = 'data/20230413'
    # label_dir = 'data/vertex'
    img, vis_img = load_img(img_name, is_raw=True)
    label = load_label(label_dir, img_name)
    return img, vis_img, label


def main(vis_edge=True, vis_point=True, args=None):
    # load model
    grid_size = 64
    timestamp = time.strftime('%Y%m%d_%H_%M_%S', time.localtime())
    save_folder = os.path.join('results', timestamp)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    data_list = read_list('test_list')
    # data_list = ['03312']
    edge_backbone, edge_model, point_backbone, point_model = None, None, None, None
    if vis_edge:
        edge_model_path = os.path.join('experiments/20230407_22_00_44', 'best.tar')
        # edge_backbone, edge_model = load_edge_model(edge_model_path)
        edge_model = load_edge_model(edge_model_path)
        print('load edge model')
    if vis_point:
        point_model_path = os.path.join('experiments/20230407_22_19_13', 'last.tar')
        point_model = load_point_model(point_model_path, args)
        print('load point model')
    corner_tp = 0
    corner_fp = 0
    corner_length = 0
    for data_name in data_list:
        # data_name = '03260'
        if data_name == '00200':
            break
        img, vis_img, merge_img, label = load_data(data_name)
        with torch.no_grad():
            if vis_edge:
                # image_feats, feat_mask, _ = edge_backbone(img)
                pixels, pixel_features = get_pixel_features(image_size=256)
                pixel_features = pixel_features.cuda()
                pixel_features = pixel_features.unsqueeze(0).repeat(img.shape[0], 1, 1, 1)
                # edge_pred, _ = edge_model(image_feats, feat_mask, pixel_features, label['grid_coords'])
                gifs_hb = edge_model(img, pixel_features, label['grid_coords'])
            if vis_point:
                # image_feats, feat_mask, all_image_feats = point_backbone(img)
                # image_feats, feat_mask, all_image_feats = point_backbone(merge_img)

                pixels, pixel_features = get_pixel_features(image_size=256)
                pixel_features = pixel_features.cuda()
                pixel_features = pixel_features.unsqueeze(0).repeat(img.shape[0], 1, 1, 1)
                pred_logits, pred_coords = point_model(img, pixel_features)
        if vis_edge:
            gifs_pred = gifs_hb.squeeze()
            gifs_pred = gifs_pred.detach().cpu().numpy()
            edge_pred = (gifs_pred >= 0.5).astype(np.float32)
            # edge_pred = np.array(edge_pred)
            p = np.array(label['grid_coords'].squeeze(0).cpu())
            for i in range(edge_pred.shape[0]):
                # if edge_pred[i][1] > edge_pred[i][0]:
                if edge_pred[i]:
                    point0, point1 = [int(p[i, 0]), int(p[i, 1])], [int(p[i, 2]), int(p[i, 3])]
                    cv2.line(vis_img, tuple(point0), tuple(point1), [0, 0, 255], 1)
        if vis_point:
            # point_pred = np.array(point_pred.squeeze(0).cpu())
            # point_idx = np.where(point_pred >= 0.2)
            # pred_confs = point_pred[point_idx]
            # pred_points = pixels[point_idx]
            # for i in range(pred_points.shape[0]):
            #     cv2.circle(vis_img, tuple(pred_points[i]), 1, [0, 255, 0], -1)
            pred_coords = np.array(pred_coords.squeeze(0).cpu())
            pred_logits = np.array(pred_logits.squeeze(0).cpu())
            # label_logits = label['vertex_flags']
            indices = np.where(pred_logits > 0)[0]
            pred_coord = pred_coords[indices]
            pred_corner = preprocess_vertex(pred_coord * 255.0, label['gt_data']['corners'])
            score = calc_corner(pred_corner, label['gt_data']['corners'])
            corner_tp += score['corner_tp']
            corner_fp += score['corner_fp']
            corner_length += score['corner_length'] # 使用grid来确定
            print(data_name)
            for i in range(pred_coord.shape[0]):
                point = [int(pred_coord[i, 0] * 255), int(pred_coord[i, 1] * 255)]
                cv2.circle(vis_img, tuple(point), 1, [0, 255, 0], -1)
            recall, precision = get_recall_and_precision(corner_tp, corner_fp, corner_length)
            f_score = 2.0 * precision * recall / (recall + precision + 1e-8)
            print('corners - precision: %.3f recall: %.3f f_score: %.3f' % (precision, recall, f_score))
        save_path = os.path.join(save_folder, data_name + '.png')
        print(save_path)
        cv2.imwrite(save_path, vis_img)
    print('finish')


def preprocess_vertex(vertexs, corners):
    pred_corners = []
    for i, corner in enumerate(corners):
        near_pred = [999999.0, (0.0, 0.0)]
        for k, vertex in enumerate(vertexs):
            dist = np.linalg.norm(corner - vertex)
            if dist < near_pred[0]:
                near_pred = [dist, vertex]
        pred_corners.append(near_pred[1])
    pred_corners = np.array(pred_corners)
    return pred_corners


def calc_corner(dets, gts, thresh=8.0):

    per_sample_corner_tp = 0.0
    per_sample_corner_fp = 0.0
    per_sample_corner_length = gts.shape[0]
    found = [False] * gts.shape[0]
    c_det_annot = {}

    # for each corner detection
    for i, det in enumerate(dets):
        # get closest gt
        near_gt = [0, 999999.0, (0.0, 0.0)]
        for k, gt in enumerate(gts):
            dist = np.linalg.norm(gt - det)
            if dist < near_gt[1]:
                near_gt = [k, dist, gt]
        if near_gt[1] <= thresh and not found[near_gt[0]]:
            per_sample_corner_tp += 1.0
            found[near_gt[0]] = True
            c_det_annot[i] = near_gt[0]
        else:
            per_sample_corner_fp += 1.0
    score = {
        'corner_tp': per_sample_corner_tp,
        'corner_fp': per_sample_corner_fp,
        'corner_length': per_sample_corner_length
    }
    return score


def calc_region(dets, gts):
    # inputs: corners ---- edge
    # 先把结果打印出来
    iou_thresh = 0.7
    size = 256
    det_mask = np.ones((2, size, size)) * 0
    corners = dets['corners']
    edges = dets['edges']
    corners = np.round(corners.copy() * 1.).astype(np.int)
    for corner_i in range(corners.shape[0]):
        det_mask[1] = cv2.circle(det_mask[1], (int(corners[corner_i, 1]), int(corners[corner_i, 0])), 3, 1.0, -1)
    for i in range(edges.shape[0]):
        point0, point1 = list(corners[edges[i][0]]), list(corners[edges[i][1]])
        if -1 in point0 or -1 in point1:
            continue
        det_mask[0] = cv2.line(det_mask[0], tuple(point0), tuple(point1), 1.0, thickness=2)
    det = det_mask[0]
    conv_mask = det_mask[0]
    conv_mask = 1 - conv_mask
    conv_mask = conv_mask.astype(np.uint8)
    labels, region_mask = cv2.connectedComponents(conv_mask, connectivity=4)
    background_label = region_mask[0, 0]
    all_conv_masks = []
    for region_i in range(1, labels):
        if region_i == background_label:
            continue
        the_region = region_mask == region_i
        if the_region.sum() < 20:
            continue
        all_conv_masks.append(the_region)

    mask = np.ones((2, size, size)) * 0
    corners = gts['corners']
    edges = gts['edges']
    corners = np.round(corners.copy() * 1.).astype(np.int)
    for corner_i in range(corners.shape[0]):
        mask[1] = cv2.circle(mask[1], (int(corners[corner_i, 1]), int(corners[corner_i, 0])), 3, 1.0, -1)
    for edge_i in range(edges.shape[0]):
        a = edges[edge_i, 0]
        b = edges[edge_i, 1]
        mask[0] = cv2.line(mask[0], (int(corners[a, 0]), int(corners[a, 1])),
                           (int(corners[b, 0]), int(corners[b, 1])), 1.0, thickness=2)
    gt = mask[0]
    gt_mask = mask[0]
    gt_mask = 1 - gt_mask
    gt_mask = gt_mask.astype(np.uint8)
    labels, region_mask = cv2.connectedComponents(gt_mask, connectivity=4)

    #cv2.imwrite('mask-gt.png', region_mask.astype(np.uint8) * 20)

    background_label = region_mask[0, 0]
    all_gt_masks = []
    for region_i in range(1, labels):
        if region_i == background_label:
            continue
        the_region = region_mask == region_i
        if the_region.sum() < 20:
            continue
        all_gt_masks.append(the_region)

    per_sample_region_tp = 0.0
    per_sample_region_fp = 0.0
    per_sample_region_length = len(all_gt_masks)
    found = [False] * len(all_gt_masks)

    for i, r_det in enumerate(all_conv_masks):
        # gt closest gt
        near_gt = [0, 0, None]
        for k, r_gt in enumerate(all_gt_masks):
            iou = np.logical_and(r_gt, r_det).sum() / float(np.logical_or(r_gt, r_det).sum())
            if iou > near_gt[1]:
                near_gt = [k, iou, r_gt]
        if near_gt[1] >= iou_thresh and not found[near_gt[0]]:
            per_sample_region_tp += 1.0
            found[near_gt[0]] = True
        else:
            per_sample_region_fp += 1.0

    return {
        'region_tp': per_sample_region_tp,
        'region_fp': per_sample_region_fp,
        'region_length': per_sample_region_length,
        'det': det,
        'gt': gt
    }


def get_recall_and_precision(tp, fp, length):
    recall = tp / (length + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    return recall, precision


def load_model(args):
    # old
    # experiments/20230324_15_21_46
    # experiments/20230328_10_14_19
    # new
    # experiments/20230407_22_00_44
    # experiments/20230407_22_19_13
    edge_model_path = os.path.join('experiments/edge', 'last.tar')
    edge_model = load_edge_model(edge_model_path)
    point_model_path = os.path.join('experiments/vertex', 'last.tar')
    point_model = load_point_model(point_model_path, args=args)
    print('load model')
    return edge_model, point_model


def get_result(edge_model, point_model, img, grid_edge):
    with torch.no_grad():
        pixels, pixel_features = get_pixel_features(image_size=256)
        pixel_features = pixel_features.cuda()
        pixel_features = pixel_features.unsqueeze(0).repeat(img.shape[0], 1, 1, 1)
        edge_pred = edge_model(img, pixel_features, grid_edge)

        pixels, pixel_features = get_pixel_features(image_size=256)
        pixel_features = pixel_features.cuda()
        pixel_features = pixel_features.unsqueeze(0).repeat(img.shape[0], 1, 1, 1)
        pred_logits, pred_coords = point_model(img, pixel_features)
    gifs_pred = edge_pred.squeeze().detach().cpu().numpy()
    edge_pred = (gifs_pred >= 0.5).astype(np.float32)

    pred_coords = np.array(pred_coords.squeeze(0).cpu())
    pred_logits = np.array(pred_logits.squeeze(0).cpu())
    # label_logits = label['vertex_flags']
    # indices = np.where(pred_logits > 0)[0]
    # pred_coord = pred_coords[indices]
    # point_idx = np.where(point_pred >= 0.5)
    # pred_confs = point_pred[point_idx]
    # pred_points = pixels[point_idx]
    return edge_pred, pred_coords, pred_logits


def generate_floor_plan(args):
    # 直接获取grid_edge
    grid_size = 32
    timestamp = time.strftime('%Y%m%d_%H_%M_%S', time.localtime())
    save_folder = os.path.join('results', 'implicit')
    data_list = read_list('test_list')
    grid_edge = get_grid_edge()
    edge_model, point_model = load_model(args)
    corner_tp, corner_fp, corner_length = 0, 0, 0
    region_tp, region_fp, region_length = 0, 0, 0
    for data_name in data_list:
        print(data_name)
        if data_name == '00010':
            break
        img, vis_img, label = load_data(data_name)
        edge_pred, pred_coord, pred_logits = get_result(edge_model, point_model, img, grid_edge)
        # print(edge_pred, point_pred)
        all_line, corners = dual_contouring(edge_pred, pred_coord, pred_logits)

        det = dict()
        det['corners'] = corners
        det['edges'] = all_line
        # score = calc_region(det, label['gt_data'])
        # region_tp += score['region_tp']
        # region_fp += score['region_fp']
        # region_length += score['region_length']
        # vis_img = np.zeros_like(vis_img, dtype=vis_img.dtype)
        score = calc_corner(corners, label['gt_data']['corners'])
        corner_tp += score['corner_tp']
        corner_fp += score['corner_fp']
        corner_length += score['corner_length']  # 使用grid来确定
        # connect_corner(all_line, corners)
        # vis_img = draw_floor_plan(vis_img, all_line, corners)
        # if not os.path.exists(save_folder):
        #     os.mkdir(save_folder)
        # save_path = os.path.join(save_folder, data_name + '.npy')
        # np.save(save_path, det)
        # print(save_path)
        # cv2.imwrite(save_path, vis_img)
    recall, precision = get_recall_and_precision(corner_tp, corner_fp, corner_length)
    # recall, precision = get_recall_and_precision(region_tp, region_fp, region_length)
    f_score = 2.0 * precision * recall / (recall + precision + 1e-8)
    print('corners - precision: %.3f recall: %.3f f_score: %.3f' % (precision, recall, f_score))


def draw_floor_plan(vis_img, all_line, corners):
    for i in range(all_line.shape[0]):
        point0, point1 = list(all_line[i][0]), list(all_line[i][1])
        if -1 in point0 or -1 in point1:
            continue
        vis_img = cv2.line(vis_img, tuple(point0), tuple(point1), [0, 0, 255], 2)
        # vis_img = cv2.circle(vis_img, tuple(point0), 1, [0, 255, 0], -1)
        # vis_img = cv2.circle(vis_img, tuple(point1), 1, [0, 255, 0], -1)
    for i in range(corners.shape[0]):
        corner = list(corners[i, :])
        vis_img = cv2.circle(vis_img, tuple(corner), 2, [255, 0, 0], -1)
    return vis_img


def dual_contouring(edge_pred, pred_coord, pred_logits):
    all_line = []
    grid_size = 32
    size_of_grid = int(256 / grid_size)
    edge_pred = edge_pred.reshape(grid_size, grid_size, 4)
    pred_logits = pred_logits.reshape(grid_size, grid_size)
    pred_coord = pred_coord.reshape(grid_size, grid_size, 2)
    grid_vertices = np.full([grid_size, grid_size, 2], -1, np.float32)
    num = 0
    corners = []
    corner_grids = []
    for i in range(grid_size):
        for j in range(grid_size):
            # 先找到corner 以及所在的网格
            flag = [0, 0, 0, 0]
            [ex1, ex2, ex3, ex4] = edge_pred[i, j, :]
            if ex1 > 0:
                flag[0] = 1
            if ex2 > 0:
                flag[1] = 1
            if ex3 > 0:
                flag[2] = 1
            if ex4 > 0:
                flag[3] = 1
            for k in range(4):
                is_corner = flag[k] + flag[(k + 1) % 4]
                if is_corner == 2:
                    corners.append(pred_coord[i, j, :] * 255.0)
                    corner_grids.append([i, j])
                    break
    # up right down left
    # find the match corners
    corner_grids = np.array(corner_grids)
    match_corners = find_connect_corners(corner_grids, edge_pred)
    for i in range(len(corners)):
        match_corner = match_corners[i]
        for corner in match_corner:
            for l, grid in enumerate(corner_grids):
                if corner[0] == grid[0] and corner[1] == grid[1]:
                    # all_line.append([corners[i], corners[l]])
                    all_line.append([i, l])
                    break
    # print(all_line)
    # for i in range(grid_size - 1):
    #     for j in range(grid_size - 1):
    #         flags = [0, 0, 0, 0]
    #         if edge_pred[i][j][0] >= 0.5:
    #             all_line.append([grid_vertices[i, j, :], grid_vertices[i - 1, j, :]])
    #             flags[0] = 1
    #         if edge_pred[i][j][1] >= 0.5:
    #             all_line.append([grid_vertices[i, j, :], grid_vertices[i + 1, j, :]])
    #             flags[1] = 1
    #         if edge_pred[i][j][2] >= 0.5:
    #             all_line.append([grid_vertices[i, j, :], grid_vertices[i, j - 1, :]])
    #             flags[2] = 1
    #         if edge_pred[i][j][3] >= 0.5:
    #             all_line.append([grid_vertices[i, j, :], grid_vertices[i, j + 1, :]])
    #             flags[3] = 1
    #         for k in range(4):
    #             is_corner = flags[k] + flags[(k + 1) % 4]
    #             if is_corner == 2:
    #                 corners.append(grid_vertices[i, j, :])
    #                 break
    return np.array(all_line, dtype=np.int64), np.array(corners, dtype=np.int64)


def find_connect_corners(corner_grids, edge_pred):
    offset = np.array([[-1, 0], [0, 1], [-1, 0], [0, -1]])
    match_corners = []
    for corner_grid in corner_grids:
        find_corners = []
        direction = edge_pred[corner_grid[0], corner_grid[1], :]
        for l, ex in enumerate(direction):
            if ex > 0:
                last_grid = corner_grid
                current_grid = last_grid + offset[l]
                match_corner = DFS(last_grid, current_grid, corner_grids, edge_pred)
                if match_corner is not None:
                    find_corners.append(match_corner)
        match_corners.append(find_corners)
    return match_corners


def DFS(last_grid, current_grid, corner_grids, edge_pred):
    offset = np.array([[-1, 0], [0, 1], [-1, 0], [0, -1]])
    for corner_grid in corner_grids:
        if current_grid[0] == corner_grid[0] and current_grid[1] == corner_grid[1]:
            return current_grid
    # if current_grid in corner_grids:
    #     return current_grid
    else:
        direction = edge_pred[current_grid[0], current_grid[1], :]
        # find_corners = []
        for l, ex in enumerate(direction):
            if ex > 0:
                next_grid = current_grid + offset[l]
                if next_grid[0] == last_grid[0] and next_grid[1] == last_grid[1]:
                    continue
                find_corner = DFS(current_grid, next_grid, corner_grids, edge_pred)
                return find_corner
        # return find_corners


def get_grid_edge():
    grid_edge = []
    grid_size = 32
    size_of_grid = int(256 / grid_size)
    for i in range(grid_size):
        for j in range(grid_size):
            # edge_x_1 = [i * size_of_grid, j * size_of_grid, i * size_of_grid, (j + 1) * size_of_grid]
            # edge_x_2 = [(i + 1) * size_of_grid, j * size_of_grid, (i + 1) * size_of_grid, (j + 1) * size_of_grid]
            # edge_y_1 = [i * size_of_grid, j * size_of_grid, (i + 1) * size_of_grid, j * size_of_grid]
            # edge_y_2 = [i * size_of_grid, (j + 1) * size_of_grid, (i + 1) * size_of_grid, (j + 1) * size_of_grid]
            # grid_edge.append([edge_x_1, edge_x_2, edge_y_1, edge_y_2])
            edge_1 = [i * size_of_grid, j * size_of_grid, i * size_of_grid, (j + 1) * size_of_grid]
            edge_2 = [i * size_of_grid, (j + 1) * size_of_grid, (i + 1) * size_of_grid, (j + 1) * size_of_grid]
            edge_3 = [(i + 1) * size_of_grid, j * size_of_grid, (i + 1) * size_of_grid, (j + 1) * size_of_grid]
            edge_4 = [i * size_of_grid, j * size_of_grid, (i + 1) * size_of_grid, j * size_of_grid]
            grid_edge.append([edge_1, edge_2, edge_3, edge_4])
    grid_edge = np.concatenate(grid_edge, axis=0, dtype=np.float32)
    grid_edge[grid_edge == 256.0] = 255.0
    grid_edge = torch.tensor(grid_edge, dtype=torch.float32).cuda().unsqueeze(0)
    return grid_edge


def load_gifs_model(model_path):
    model = GIFS_Former(input_dim=128, hidden_dim=256)
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['model_state_dict'])
    return model


def generate_floor_plan_v1(args):
    timestamp = time.strftime('%Y%m%d_%H_%M_%S', time.localtime())
    save_folder = os.path.join('results', 'coco_vertex_wo_aug' + '_floor_plan')
    data_list = read_list('test_list')
    model_path = os.path.join('/mnt/d/sdb/honghao/heat/experiments/coco_vertex_wo_aug_w_raster_loss_w_gifs_loss', 'last.tar')
    model = load_gifs_model(model_path)
    # vertex_model_path = os.path.join('experiments/vertex', 'last.tar')
    # vertex_model = load_point_model(vertex_model_path, args)
    grid_edge_p = get_grid_edge()
    sum_f1 = 0
    # 1: predict vertex
    # 2: choose corner
    # 3: connect corner
    # data_list = ['00000']
    for data_name in data_list:
        # if data_name == '03251':
        #     break
        annot_folder = 'data/s3d_floorplan/annot'
        annot_path = os.path.join(annot_folder, data_name + '.npy')
        annot = np.load(annot_path, allow_pickle=True, encoding='latin1').tolist()
        gt_data = convert_annot(annot)
        img, vis_img, label = load_data(data_name)
        p = torch.tensor(label.get('coords')).unsqueeze(0).cuda()
        # p = grid_edge
        with torch.no_grad():
            pixels, pixel_features = get_pixel_features(image_size=256)
            pixel_features = pixel_features.cuda()
            pixel_features = pixel_features.unsqueeze(0).repeat(img.shape[0], 1, 1, 1)
            gifs, pred_coords, pred_logits = model(img, pixel_features, p)
            # pixels, pixel_features = get_pixel_features(image_size=256)
            # pixel_features = pixel_features.cuda()
            # pixel_features = pixel_features.unsqueeze(0).repeat(img.shape[0], 1, 1, 1)
            # gifs, pred_coords, pred_logits = model(img, pixel_features, p)
        pred_gt = (torch.sigmoid(gifs) > 0.5).squeeze().detach().cpu().numpy().astype(np.float32)
        p = np.array(p.squeeze(0).cpu())
        for i in range(pred_gt.shape[0]):
            if pred_gt[i]:
                point0, point1 = [int(p[i, 0]), int(p[i, 1])], [int(p[i, 2]), int(p[i, 3])]
                cv2.line(vis_img, tuple(point0), tuple(point1), [0, 0, 255], 1)
            # else:
            #     point0, point1 = [int(p[i, 0]), int(p[i, 1])], [int(p[i, 2]), int(p[i, 3])]
            #     cv2.line(vis_img, tuple(point0), tuple(point1), [0, 255, 0], 1)
        # pred_coords = np.array(pred_coords.squeeze(0).cpu())
        # # pred_logits = np.array(pred_logits.squeeze(0).cpu())
        # pred_logits = (torch.sigmoid(pred_logits) > 0.5).squeeze().detach().cpu().numpy().astype(np.float32)
        # # label_logits = label['vertex_flags']
        # indices = np.where(pred_logits > 0)[0]
        # pred_coord = pred_coords[indices]
        # grid_edge = []
        # radius = 2
        # for single_coord in pred_coord:
        #     single_coord = single_coord * 255.0
        #     [x, y] = single_coord
        #     # grid_edge.append([x - radius, y - grid_length, x - grid_length, y - radius])
        #     # grid_edge.append([x - grid_length, y - radius, x + grid_length, y - radius])
        #     # grid_edge.append([x + grid_length, y - radius, x + radius, y - grid_length])
        #     # grid_edge.append([x + radius, y - grid_length, x + radius, y + grid_length])
        #     # grid_edge.append([x + radius, y + grid_length, x + grid_length, y + radius])
        #     # grid_edge.append([x + grid_length, y + radius, x - grid_length, y + radius])
        #     # grid_edge.append([x - grid_length, y + radius, x - radius, y + grid_length])
        #     # grid_edge.append([x - radius, y + grid_length, x - radius, y - grid_length])
        #     x1, x2 = single_coord[0] - radius, single_coord[0] + radius
        #     y1, y2 = single_coord[1] - radius, single_coord[1] + radius
        #     grid_edge.append([x1, y1, x1, y2])
        #     grid_edge.append([x1, y1, x2, y1])
        #     grid_edge.append([x2, y1, x2, y2])
        #     grid_edge.append([x1, y2, x2, y2])
        # grid_edge = np.array(grid_edge, dtype=np.float32)
        # grid_edge[grid_edge == 256.0] = 255.0
        # p = torch.tensor(grid_edge, dtype=torch.float32).cuda().unsqueeze(0)
        # with torch.no_grad():
        #     pixels, pixel_features = get_pixel_features(image_size=256)
        #     pixel_features = pixel_features.cuda()
        #     pixel_features = pixel_features.unsqueeze(0).repeat(img.shape[0], 1, 1, 1)
        #     gifs, pred_coords, pred_logits = model(img, pixel_features, p)
        # pred_gt = (torch.sigmoid(gifs) > 0.5).squeeze().detach().cpu().numpy().astype(np.float32)
        # pred_gt = pred_gt.reshape(pred_coord.shape[0], 4)
        # # choose corner
        # corners = []
        # for i in range(pred_gt.shape[0]):
        #     pred = pred_gt[i]
        #     flag = [0, 0, 0, 0]
        #     [ex1, ex2, ex3, ex4] = pred
        #     if ex1 > 0:
        #         flag[0] = 1
        #     if ex2 > 0:
        #         flag[1] = 1
        #     if ex3 > 0:
        #         flag[2] = 1
        #     if ex4 > 0:
        #         flag[3] = 1
        #     for k in range(4):
        #         is_corner = flag[k] + flag[(k + 1) % 4]
        #         if is_corner == 2:
        #             corners.append(pred_coord[i, :] * 255.0)
        #             break
        # corners = np.array(corners, dtype=np.int64)
        # for i in range(corners.shape[0]):
        #     corner = corners[i]
        #     cv2.circle(vis_img, tuple(corner), 2, [0, 0, 255], -1)
        # # for i in range(pred_gt.shape[0]):
        # #     if pred_gt[i]:
        # #         point0, point1 = [int(p[i, 0]), int(p[i, 1])], [int(p[i, 2]), int(p[i, 3])]
        # #         cv2.line(vis_img, tuple(point0), tuple(point1), [0, 0, 255], 1)
        # # connect corner
        # pixels, pixel_features = get_pixel_features(image_size=256)
        # pixel_features = pixel_features.cuda()
        # pixel_features = pixel_features.unsqueeze(0).repeat(img.shape[0], 1, 1, 1)
        # # gifs, pred_coords, pred_logits = model(img, pixel_features, p)
        # all_line = []
        # corner_num = 3
        # for id1, point1 in enumerate(corners):
        #     num = 0
        #     for id2, point2 in enumerate(corners):
        #         if id1 == id2:
        #             continue
        #         if num > 3:
        #             break
        #         # judge edge
        #         sample_points = sample_pair(point1, point2)
        #         p = torch.tensor(sample_points).unsqueeze(0).cuda()
        #         gifs, pred_coords, pred_logits = model(img, pixel_features, p)
        #         pred_gt = (torch.sigmoid(gifs) > 0.5).squeeze().detach().cpu().numpy().astype(np.float32)
        #         if pred_gt.sum() == pred_gt.shape[0]:
        #             num += 1
        #             all_line.append([point1, point2])
        #             for pair in sample_points:
        #                 pt1, pt2 = pair[0:2], pair[2:4]
        #                 a = tuple([int(pt1[0]), int(pt1[1])])
        #                 b = tuple([int(pt2[0]), int(pt2[1])])
        #                 cv2.line(vis_img, a, b, [0, 0, 255], 1)
        #                 cv2.circle(vis_img, a, 1, [0, 255, 0], -1)
        #                 cv2.circle(vis_img, b, 1, [0, 255, 0], -1)
        #             print('add line')
        #             print(id1, id2)
        # #         # judge implicit connection
        # # # for i in range(pred_gt.shape[0]):
        # # #     if pred_gt[i]:
        # # #         point0, point1 = [int(p[i, 0]), int(p[i, 1])], [int(p[i, 2]), int(p[i, 3])]
        # # #         cv2.line(vis_img, tuple(point0), tuple(point1), [0, 0, 255], 1)
        # corners = gt_data['corners'].astype(np.int64)
        # for i in range(len(corners)):
        #     cv2.circle(vis_img, tuple(corners[i, :]), 3, [0, 0, 255], -1)
        # for i in range(pred_coord.shape[0]):
        #     point = [int(pred_coord[i, 0] * 255), int(pred_coord[i, 1] * 255)]
        #     cv2.circle(vis_img, tuple(point), 1, [0, 255, 0], -1)
        # for line in all_line:
        #     cv2.line(vis_img, tuple(line[0]), tuple(line[1]), [0, 0, 255], 1)
        save_path = os.path.join(save_folder, data_name + '.png')
        print(save_path)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        cv2.imwrite(save_path, vis_img)
        print(save_path)
    #     gt = label.get('flags')
    #     gt = np.squeeze(gt, axis=1)
    #     pos_gt_ids = np.where(gt == 1)
    #     correct = (pred_gt[pos_gt_ids] == gt[pos_gt_ids]).astype(np.float32).sum()
    #     recall = correct / len(pos_gt_ids[0])
    #     num_pred_pos = (pred_gt == 1).astype(np.float32).sum()
    #     prec = correct / num_pred_pos
    #     f_score = 2.0 * prec * recall / (recall + prec + 1e-8)
    #     sum_f1 += f_score
    #     print(data_name, f_score)
    # f1 = sum_f1 / len(data_list)
    # print('f1: ', f1)

def sample_pair(point1, point2):
    sample_num = 10
    offset = 2
    num = sample_num + 1
    line_dist = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)  # 线段长度
    line = [point1[0], point1[1], point2[0], point2[1]]  # 两点成线
    line_ABC = line_general_equation(line)  # 一般式规范化
    newP = []
    # newP.append(point1.tolist())  # 压入首端点
    if num > 0:
        dxy = line_dist / num  # 实际采样距离
        # ic(dxy)
        for i in range(1, num):
            if line_ABC[1] != 0:
                alpha = np.arctan(-line_ABC[0])
                dx = dxy * np.cos(alpha)
                dy = dxy * np.sin(alpha)
                if point2[0] - point1[0] > 0:
                    newP.append([point1[0] + i * dx, point1[1] + i * dy])
                else:
                    newP.append([point1[0] - i * dx, point1[1] - i * dy])
            else:
                if point2[1] - point1[1] > 0:
                    newP.append([point1[0], point1[1] + i * dxy])
                else:
                    newP.append([point1[0], point1[1] - i * dxy])
    # newP.append([point2[0], point2[1]])  # 压入末端点
    tangent = point2 - point1
    tangent = tangent / np.linalg.norm(tangent, 2)
    normal = np.array([-1 * tangent[1], tangent[0]])
    newP = np.array(newP)
    result = []
    for point in newP:
        new_point1 = point + normal * offset
        new_point2 = point - normal * offset
        result.append([new_point1, new_point2])
    result = np.array(result)
    result = result.reshape(result.shape[0], 4)
    return result

def line_general_equation(line):
    """直线一般式"""
    A = line[3] - line[1]
    B = line[0] - line[2]
    C = line[2] * line[1] - line[0] * line[3]
    line = np.array([A, B, C])
    if B != 0:
        line = line / B
    return line


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RoomFormer training script', parents=[get_args_parser()])
    args = parser.parse_args()
    # main(vis_edge=False, vis_point=True, args=args)
    # generate_floor_plan(args=args)
    generate_floor_plan_v1(args)
    print('finish')
