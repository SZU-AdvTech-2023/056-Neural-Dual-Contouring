# TODO: 对比两种预测vertex的方式，计算三个指标
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import floor_data
from models.GIFS_HEAT import GIFS_HEAT, GIFS_HEATv1, GIFS_HEATv2
from datasets.data_utils import collate_fn, get_pixel_features
from tqdm import tqdm
import wandb
import math
import argparse


def main(args):
    wandb.init(project="ablation_vertex")
    print(args)
    exp_name = 'no_attention'
    exp_dir = os.path.join('experiments', exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
    if not args.using_attention:
        model = GIFS_HEATv2(input_dim=128, hidden_dim=256)
    train_dataset = floor_data.FloorPlanDataset(
        'train',
        data_dir='data/20230413',
        is_raw=True,
        rand_aug=False,
    )
    val_dataset = floor_data.FloorPlanDataset(
        'val',
        data_dir='data/20230413',
        is_raw=True,
        rand_aug=False,
    )
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    params = [p for p in model.parameters()]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    epochs = args.epochs
    max_norm = args.clip_max_norm

    # training
    model = nn.DataParallel(model)
    model = model.cuda()
    f_max = -1e9
    for epoch in range(epochs):
        model.train()
        sum_loss = 0
        print('epoch:', epoch)
        for batch in tqdm(train_data_loader):
            optimizer.zero_grad()
            inputs = batch.get('inputs').cuda()
            # image_feats, feat_mask, all_image_feats = self.backbone(inputs)
            pixels, pixel_features = get_pixel_features(image_size=256)
            pixel_features = pixel_features.cuda()
            pixel_features = pixel_features.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
            point_pred = model(inputs, pixel_features, pixels)
            vertex_map = batch.get('vertex_map').cuda()
            # loss = F.binary_cross_entropy(point_pred.double(), vertex_map.double())
            loss = torch.nn.BCELoss()(point_pred.double(), vertex_map.double())
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            sum_loss += loss
            wandb.log({"train_loss_last_batch": loss, "epoch": epoch})
        lr_scheduler.step()
        print('sum loss: ', sum_loss)
        wandb.log({"train_loss_avg": sum_loss / len(train_data_loader), "epoch": epoch})
        model.eval()
        sum_f1 = 0
        sum_prec = 0
        sum_recall = 0
        for val_batch in val_data_loader:
            inputs = val_batch.get('inputs').cuda()
            with torch.no_grad():
                pixels, pixel_features = get_pixel_features(image_size=256)
                pixel_features = pixel_features.cuda()
                pixel_features = pixel_features.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
                point_pred = model(inputs, pixel_features, pixels)
            point_pred = (point_pred > 0.5).squeeze().detach().cpu().numpy().astype(np.float32)
            label = val_batch.get('vertex_map').squeeze().detach().cpu().numpy()
            pos_gt_ids = np.where(label == 1)
            correct = (point_pred[pos_gt_ids] == label[pos_gt_ids]).astype(np.float32).sum()
            recall = correct / len(pos_gt_ids[0])
            num_pred_pos = (point_pred == 1).astype(np.float32).sum()
            prec = correct / num_pred_pos
            f_score = 2.0 * prec * recall / (recall + prec + 1e-8)
            sum_f1 += f_score
            sum_prec += prec
            sum_recall += recall
        f1 = sum_f1 / len(val_data_loader)
        prec = sum_prec / len(val_data_loader)
        recall = sum_recall / len(val_data_loader)
        print(f'acc:{f1}, prec:{prec}, recall:{recall}')
        wandb.log({"acc": f1, "epoch": epoch})
        wandb.log({"prec": prec, "epoch": epoch})
        wandb.log({"recall": recall, "epoch": epoch})
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
                save_path, )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", action="store", dest="lr", default=1e-4, type=float, help="Learning rate [0.0001]")
    parser.add_argument("--epochs", action="store", dest="epochs", default=300, type=int, help="epochs")
    parser.add_argument("--batch_size", action="store", dest="batch_size", default=8, type=int, help="batch_size")
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    parser.add_argument('--using_attention', default=False, type=bool)
    # parser.add_argument('--lr_drop', default=300, type=int)
    FLAGS = parser.parse_args()
    main(FLAGS)
