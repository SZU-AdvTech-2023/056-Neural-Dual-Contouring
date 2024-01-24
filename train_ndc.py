import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import floor_data
from models.resnet import ResNetBackbone
from models.GIFS_HEAT import GIFS_HEAT, GIFS_HEATv1
from models.loss import sigmoid_focal_loss
from datasets.data_utils import collate_fn, get_pixel_features
from tqdm import tqdm
import wandb
import math
import argparse


class Runner:
    def __init__(self, args):
        self.args = args
        timestamp = time.strftime('%Y%m%d_%H_%M_%S', time.localtime())
        self.exp_dir = os.path.join('experiments', timestamp)
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.edge_model = GIFS_HEATv1(input_dim=128, hidden_dim=256)
        self.device = torch.device("cuda:"+str(args.gpu))
        self.train_dataset = floor_data.FloorPlanDataset(
            "train",
            data_dir=args.data_dir,
            is_raw=args.is_raw,
            rand_aug=args.aug,
            is_point=False,
            is_edge=True
        )
        self.val_dataset = floor_data.FloorPlanDataset(
            "val",
            data_dir=args.data_dir,
            is_raw=args.is_raw,
            rand_aug=False,
            is_point=False,
            is_edge=True
        )
        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)
        self.lr = args.lr
        all_params = [p for p in self.edge_model.parameters()]
        # all_params = backbone_params + edge_params
        # self.optimizer = optim.Adam(all_params, lr=self.lr)
        self.optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, args.lr_drop)
        self.epochs = args.epochs
        self.max_norm = args.clip_max_norm
        self.edge_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).cuda(), reduction='mean')

    def train(self):
        wandb.init(project="0404_gifs")
        # self.backbone = nn.DataParallel(self.backbone)
        self.edge_model = nn.DataParallel(self.edge_model)
        # self.backbone = self.backbone.to(self.device)
        # self.edge_model = self.edge_model.to(self.device)
        # self.backbone = self.backbone.cuda()
        self.edge_model = self.edge_model.cuda()
        val_min = 1e8
        f1_max = -1
        for epoch in range(self.epochs):
            self.edge_model.train()
            sum_loss = 0
            print('epoch:', epoch)
            for batch in tqdm(self.train_data_loader):
                self.optimizer.zero_grad()
                p = batch.get('grid_coords')
                # inputs = batch.get('inputs').to(self.device)
                inputs = batch.get('inputs').cuda()
                # image_feats, feat_mask, _ = self.backbone(inputs)
                pixels, pixel_features = get_pixel_features(image_size=256)
                # pixel_features = pixel_features.to(self.device)
                pixel_features = pixel_features.cuda()
                pixel_features = pixel_features.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
                gifs_hb = self.train_batch(inputs, pixel_features, p)
                # label = batch.get('labels').to(self.device)
                # mask = batch.get('mask').cuda()
                label = batch.get('labels').cuda()
                # loss = self.edge_loss(gifs_hb, label.squeeze(1))
                loss = F.binary_cross_entropy_with_logits(gifs_hb, label)
                loss.backward()
                # print(loss.item())
                if self.max_norm > 0:
                    # torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), self.max_norm)
                    torch.nn.utils.clip_grad_norm_(self.edge_model.parameters(), self.max_norm)
            #
                self.optimizer.step()
            #     # self.optimizer.step()
                sum_loss += loss
            #     # sum_hb_loss += hb_loss
            #     # sum_rel_loss += rel_loss
                wandb.log({"train_loss_last_batch": loss, "epoch": epoch})
            self.lr_scheduler.step()
            print('sum loss: ', sum_loss)
            # # # prec = sum_correct / sum_prec
            # # # recall = sum_correct / sum_recall
            # # # acc = 2.0 * prec * recall / (recall + prec + 1e-8)
            # # # print(prec, recall, acc)
            wandb.log({"train_loss_avg": sum_loss / len(self.train_data_loader), "epoch": epoch})
            # # wandb.log({"hb_loss": sum_hb_loss / len(self.train_data_loader), "epoch": epoch})
            # # wandb.log({"rel_loss": sum_rel_loss / len(self.train_data_loader), "epoch": epoch})
            # # self.backbone.eval()
            self.edge_model.eval()
            # sum_val_loss = 0
            # sum_correct = 0
            # sum_down = 0
            sum_f1 = 0
            for val_batch in self.val_data_loader:
                p = val_batch.get('grid_coords')
                # inputs = val_batch.get('inputs').to(self.device)
                inputs = val_batch.get('inputs').cuda()
                with torch.no_grad():
                    # image_feats, feat_mask, _ = self.backbone(inputs)
                    pixels, pixel_features = get_pixel_features(image_size=256)
                    # pixel_features = pixel_features.to(self.device)
                    pixel_features = pixel_features.cuda()
                    pixel_features = pixel_features.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
                    gifs_hb = self.train_batch(inputs, pixel_features, p)
                pred_gt = (torch.sigmoid(gifs_hb) > 0.5).squeeze().detach().cpu().numpy().astype(np.float32)
                # gifs_pred = gifs_pred[1, :].detach().cpu().numpy()
                label = val_batch.get('labels').squeeze().detach().cpu().numpy()
                # pred_gt = (gifs_pred >= 0.5).astype(np.float32)
                pos_gt_ids = np.where(label == 1)
                correct = (pred_gt[pos_gt_ids] == label[pos_gt_ids]).astype(np.float32).sum()
                recall = correct / len(pos_gt_ids[0])
                num_pred_pos = (pred_gt == 1).astype(np.float32).sum()
                prec = correct / num_pred_pos
                f_score = 2.0 * prec * recall / (recall + prec + 1e-8)
                sum_f1 += f_score
                # acc = self.calc_acc(gifs_pred, label)
                # print('1')
                # sum_val_loss += loss.item()
            f1 = sum_f1 / len(self.val_data_loader)
            print('acc:', f1)
            # print('recall:', sum_correct / sum_down)
            wandb.log({"acc": f1, "epoch": epoch})
            save_path = os.path.join(self.exp_dir, 'last.tar')
            print('save checkpoints: ', save_path)
            torch.save(
                {
                    "epoch": epoch,
                    # "backbone_state_dict": self.backbone.state_dict(),
                    "edge_model_state_dict": self.edge_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "args": self.args
                },
                save_path,
            )
            if f1 > f1_max:
                f1_max = f1
                # print('val_min: ', val_min)
                save_path = os.path.join(self.exp_dir, 'best.tar')
                print('save checkpoints: ', save_path)
                torch.save(
                    {
                        "epoch": epoch,
                        # "backbone_state_dict": self.backbone.state_dict(),
                        "edge_model_state_dict": self.edge_model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "args": self.args
                    },
                    save_path,
                )

    def calc_acc(self, pred, gt):
        pred_gt = (pred >= 0.5).astype(np.float32)
        pos_gt_ids = np.where(gt == 1)
        correct = (pred_gt[pos_gt_ids] == gt[pos_gt_ids]).astype(np.float32).sum()
        recall = correct / len(pos_gt_ids[0])
        num_pred_pos = (pred_gt == 1).astype(np.float32).sum()
        prec = correct / num_pred_pos
        f_score = 2.0 * prec * recall / (recall + prec + 1e-8)
        return f_score

    def train_batch(self, inputs, pixel_features, p):
        gifs_hb = self.edge_model(inputs, pixel_features, p.cuda())
        return gifs_hb

    def update_learning_rate(self, epoch):
        warn_up = 10
        max_iter = self.epochs
        init_lr = self.lr
        lr = (epoch / warn_up) if epoch < warn_up else 0.5 * (
                    math.cos((epoch - warn_up) / (max_iter - warn_up) * math.pi) + 1)
        lr = lr * init_lr

        for g in self.optimizer.param_groups:
            g['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", action="store", dest="lr", default=1e-4, type=float, help="Learning rate [0.0001]")
    parser.add_argument("--gpu", action="store", dest="gpu", default="0", help="to use which GPU [0]")
    parser.add_argument("--data_dir", action="store", dest="data_dir", default='data/20230322_ndc', type=str)
    parser.add_argument("--epochs", action="store", dest="epochs", default=500, type=int, help="epochs")
    parser.add_argument("--batch_size", action="store", dest="batch_size", default=4, type=int, help="batch_size")
    parser.add_argument("--is_raw", action="store", dest="is_raw", default=True, type=bool, help="is_raw")
    parser.add_argument("--aug", action="store", dest="aug", default=True, type=bool, help="aug")
    parser.add_argument('--lr_drop', default=300, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    FLAGS = parser.parse_args()
    print(FLAGS)
    runner = Runner(FLAGS)
    runner.train()
