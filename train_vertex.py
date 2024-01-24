import argparse
import datetime
import json
import random
import os
import time
from pathlib import Path
import torch.nn as nn
import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader
from models.VertexFormer import VertexModel
from floor_data import vertex_dataset
from floor_data import FloorPlanDataset
from datasets.data_utils import collate_fn, get_pixel_features
import torch.nn.functional as F
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser('Vertex', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--lr_drop', default=[300], type=list)
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
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--job_name', default='train_stru3d', type=str)

    return parser


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


def get_recall_and_precision(tp, fp, length):
    recall = tp / (length + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    return recall, precision


def main(args):
    # print("git:\n  {}\n".format(utils.get_sha()))
    timestamp = time.strftime('%Y%m%d_%H_%M_%S', time.localtime())
    exp_dir = os.path.join('experiments', timestamp)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = VertexModel(input_dim=128, args=args)
    # model.to(device)
    model = nn.DataParallel(model)
    model.cuda()

    params = [p for p in model.parameters()]
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('number of params:', n_parameters)

    # build dataset and dataloader
    train_dataset = FloorPlanDataset(
        "train",
        data_dir='data/20230410',
        is_raw=True,
        rand_aug=True,
    )
    val_dataset = FloorPlanDataset(
        "val",
        data_dir='data/20230410',
        is_raw=True,
        rand_aug=False,
    )
    # train_dataset = vertex_dataset(
    #         "train",
    #         data_dir='data/vertex',
    #     )
    # val_dataset = vertex_dataset(
    #         "val",
    #         data_dir='data/vertex',
    #     )
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    print("Start training")
    wandb.init(project="0403point")
    val_min = 1e8
    f_max = -1
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        sum_loss = 0
        sum_cls_loss = 0
        sum_coord_loss = 0
        print('epoch:', epoch)
        for batch in tqdm(train_data_loader):
            optimizer.zero_grad()
            inputs = batch.get('inputs').cuda()
            # vertex_center = batch.get('vertex_centers').cuda()
            pixels, pixel_features = get_pixel_features(image_size=256)
            pixel_features = pixel_features.cuda()
            pixel_features = pixel_features.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
            pred_logits, pred_coords = model(inputs, pixel_features)
            label_logits = batch.get('corner_flags').cuda()
            label_coords = batch.get('corner_coords').cuda()
            loss_ce = F.binary_cross_entropy_with_logits(pred_logits, label_logits)
            bs = label_logits.shape[0]
            total_loss = 0
            # indice_len = 0
            for b_i in range(bs):
                indices = torch.where(label_logits[b_i, :, :] > 0)[0]
                pred_coord = pred_coords[b_i, :, :][indices]
                label_coord = label_coords[b_i, :, :][indices]
                # total_loss += torch.cdist(pred_coord, label_coord, p=1).mean()
                total_loss += torch.nn.L1Loss()(pred_coord, label_coord)
                # indice_len += indices.shape[0]
            # loss_coord = total_loss / bs
                # compute loss
            # loss_coord = total_loss
            # indices = np.where(label_logit > 0)
            # loss_coords = torch.nn.L1Loss()(pred_coords * mask, label_coords * mask)
            # loss = loss_ce + loss_coord * 5
            loss = total_loss + loss_ce
            sum_cls_loss += loss_ce
            sum_coord_loss += total_loss
            # print(loss.item())
            loss.backward()
            optimizer.step()
            sum_loss += loss
            wandb.log({"train_loss_last_batch": loss, "epoch": epoch})
        print('train_loss:', sum_loss)
        lr_scheduler.step()
        wandb.log({"train_loss_avg": sum_loss / len(train_data_loader), "epoch": epoch})
        wandb.log({"cls_loss_avg": sum_cls_loss / len(train_data_loader), "epoch": epoch})
        wandb.log({"coord_loss_avg": sum_coord_loss / len(train_data_loader), "epoch": epoch})

        model.eval()
        sum_val_loss = 0
        corner_tp, corner_fp, corner_length = 0, 0, 0
        for val_batch in val_data_loader:
            inputs = val_batch.get('inputs').cuda()
            with torch.no_grad():
                # vertex_center = val_batch.get('vertex_centers').cuda()
                pixels, pixel_features = get_pixel_features(image_size=256)
                pixel_features = pixel_features.cuda()
                pixel_features = pixel_features.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
                pred_logits, pred_coords = model(inputs, pixel_features)

                pred_coords = np.array(pred_coords.squeeze(0).cpu())
                pred_logits = np.array(pred_logits.squeeze(0).cpu())
                # label_logits = label['vertex_flags']
                indices = np.where(pred_logits > 0)[0]
                pred_corners = pred_coords[indices]

            label_logits = val_batch.get('corner_flags').cuda()
            label_coords = val_batch.get('corner_coords').cuda()
            corner_logits_gt = np.array(label_logits.squeeze(0).cpu())
            corner_coords_gt = np.array(label_coords.squeeze(0).cpu())
            indices = np.where(corner_logits_gt > 0)[0]
            gt_corners = corner_coords_gt[indices]
            score = calc_corner(pred_corners, gt_corners)
            corner_tp += score['corner_tp']
            corner_fp += score['corner_fp']
            corner_length += score['corner_length']
        recall, precision = get_recall_and_precision(corner_tp, corner_fp, corner_length)
        f_score = 2.0 * precision * recall / (recall + precision + 1e-8)
        print('acc:', f_score)
        wandb.log({"acc": f_score, "epoch": epoch})
        save_path = os.path.join(exp_dir, 'last.tar')
        print('save checkpoints: ', save_path)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": args
            },
            save_path,
        )
        if f_score > f_max:
            f_max = f_score
        # if sum_val_loss < val_min:
        #     val_min = sum_val_loss
            print('f_1 max: ', f_max)
            save_path = os.path.join(exp_dir, 'best.tar')
            print('save checkpoints: ', save_path)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": args
                },
                save_path,)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RoomFormer training script', parents=[get_args_parser()])
    args = parser.parse_args()
    now = datetime.datetime.now()
    run_id = now.strftime("%Y-%m-%d-%H-%M-%S")
    args.run_name = run_id + '_' + args.job_name
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
