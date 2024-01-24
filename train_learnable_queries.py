import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'

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
# from models.GIFS_Former import LearnableFormer
from models.LearnableTransformer import Learnable_Former
from floor_data import FloorPlanDataset
from datasets.data_utils import collate_fn, get_pixel_features
import torch.nn.functional as F
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser('Vertex', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--lr_drop', default=[300], type=list)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--resume', default=False, type=bool)
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
    exp_dir = os.path.join('/mnt/d/sdb/honghao/heat/experiments', 'implicit_field_with_learnable_queries_deca')
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
    print(args)
    start_epoch = 0

    model = Learnable_Former(input_dim=128, hidden_dim=256)
    # model.to(device)
    model = nn.DataParallel(model)
    model.cuda()

    params = [p for p in model.parameters()]

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    max_norm = args.clip_max_norm
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    train_dataset = FloorPlanDataset(
        "train",
        data_dir='data/vertex_32',
        rand_aug=False,
    )
    val_dataset = FloorPlanDataset(
        "val",
        data_dir='data/vertex_32',
        rand_aug=False,
    )

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print("Start training")
    wandb.init(project="implicit_field")
    val_min = 1e8
    f_max = -1
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        sum_loss = 0
        sum_cls_loss = 0
        sum_coord_loss = 0
        sum_gifs_loss = 0
        print('epoch:', epoch)
        for batch in tqdm(train_data_loader):
            optimizer.zero_grad()
            inputs = batch.get('inputs').cuda()
            coords = batch.get('coords').cuda()
            flags = batch.get('flags').cuda()
            # vertex_center = batch.get('vertex_centers').cuda()
            pixels, pixel_features = get_pixel_features(image_size=256)
            # 每个pixel --- positional encoding
            pixel_features = pixel_features.cuda()
            pixel_features = pixel_features.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
            # 按照 之前的方法训练vertex --- 返回值
            # without matching
            gifs, pred_coords, pred_logits = model(inputs, pixel_features, coords)
            # gifs, pred_coords, pred_logits = model(inputs, pixel_features, coords)
            label_logits = batch.get('vertex_flags').cuda()
            label_coords = batch.get('vertex_coords').cuda()
            loss_ce = F.binary_cross_entropy_with_logits(pred_logits, label_logits)
            # # loss_gifs = F.binary_cross_entropy_with_logits(gifs, flags)
            bs = label_logits.shape[0]
            loss_coords = 0
            # # indice_len = 0
            for b_i in range(bs):
                indices = torch.where(label_logits[b_i, :, :] > 0)[0]
                pred_coord = pred_coords[b_i, :, :][indices]
                label_coord = label_coords[b_i, :, :][indices]
                # total_loss += torch.cdist(pred_coord, label_coord, p=1).mean()
                loss_coords += torch.nn.L1Loss()(pred_coord, label_coord)
                # indice_len += indices.shape[0]
            loss_gifs = F.binary_cross_entropy_with_logits(gifs, flags)
            loss = loss_gifs + (loss_ce + loss_coords) * 2
            sum_gifs_loss += loss_gifs
            sum_cls_loss += loss_ce
            sum_coord_loss += loss_coords
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            sum_loss += loss
            wandb.log({"train_loss_last_batch": loss, "epoch": epoch})
        print('train_loss:', sum_loss)
        lr_scheduler.step()
        wandb.log({"train_loss_avg": sum_loss / len(train_data_loader), "epoch": epoch})
        wandb.log({"cls_loss_avg": sum_cls_loss / len(train_data_loader), "epoch": epoch})
        wandb.log({"coord_loss_avg": sum_coord_loss / len(train_data_loader), "epoch": epoch})
        wandb.log({"gifs_loss_avg": sum_gifs_loss / len(train_data_loader), "epoch": epoch})
        model.eval()
        sum_f1 = 0
        sum_vertex_f1 = 0
        sum_prec = 0
        sum_recall = 0
        for val_batch in val_data_loader:
            inputs = val_batch.get('inputs').cuda()
            p = val_batch.get('coords').cuda()
            with torch.no_grad():
                # vertex_center = val_batch.get('vertex_centers').cuda()
                pixels, pixel_features = get_pixel_features(image_size=256)
                pixel_features = pixel_features.cuda()
                pixel_features = pixel_features.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
                # gifs, pred_coords, pred_logits = model(inputs, pixel_features, p)
                gifs, pred_coords, pred_logits = model(inputs, pixel_features, p)
            pred_gt = (torch.sigmoid(gifs) > 0.5).squeeze().detach().cpu().numpy().astype(np.float32)
            # gifs_pred = gifs_pred[1, :].detach().cpu().numpy()
            label = val_batch.get('flags').squeeze().detach().cpu().numpy()
            # pred_gt = (gifs_pred >= 0.5).astype(np.float32)
            pos_gt_ids = np.where(label == 1)
            correct = (pred_gt[pos_gt_ids] == label[pos_gt_ids]).astype(np.float32).sum()
            recall = correct / len(pos_gt_ids[0])
            num_pred_pos = (pred_gt == 1).astype(np.float32).sum()
            prec = correct / num_pred_pos
            f_score = 2.0 * prec * recall / (recall + prec + 1e-8)
            sum_f1 += f_score

            pred_logits = (torch.sigmoid(pred_logits) > 0.5).squeeze().detach().cpu().numpy().astype(np.float32)
            label_logits = val_batch.get('vertex_flags').squeeze().detach().cpu().numpy()
            pos_gt_ids = np.where(label_logits == 1)
            correct = (pred_logits[pos_gt_ids] == label_logits[pos_gt_ids]).astype(np.float32).sum()
            recall = correct / len(pos_gt_ids[0])
            num_pred_pos = (pred_logits == 1).astype(np.float32).sum()
            prec = correct / num_pred_pos
            f_score = 2.0 * prec * recall / (recall + prec + 1e-8)
            sum_vertex_f1 += f_score
            sum_prec += prec
            sum_recall += recall

        f1 = sum_f1 / len(val_data_loader)
        f1_vertex = sum_vertex_f1 / len(val_data_loader)
        # prec = sum_prec / len(val_data_loader)
        # recall = sum_recall / len(val_data_loader)
        print('acc:', f1)
        print(f'vertex_acc:{f1_vertex}')
        wandb.log({"acc": f1, "epoch": epoch})
        wandb.log({"vertex_acc": f1_vertex, "epoch": epoch})
        # wandb.log({"vertex_prec": prec, "epoch": epoch})
        # wandb.log({"vertex_recall": recall, "epoch": epoch})
        # wandb.log({"prec": prec, "epoch": epoch})
        # wandb.log({"recall": recall, "epoch": epoch})
        save_path = os.path.join(exp_dir, 'last.tar')
        print('save checkpoints: ', save_path)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args
            },
            save_path,
        )
        if f1 > f_max:
            f_max = f1
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
                    "lr_scheduler": lr_scheduler.state_dict(),
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
    main(args)
