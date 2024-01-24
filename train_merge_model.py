import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import floor_data
from models.resnet import ResNetBackbone
from models.GIFS_HEAT import GIFS_HEAT, GIFS_HEATv1, GIFS_HEATv2, GIFS_HEATv3
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
        self.backbone = ResNetBackbone()
        self.strides = self.backbone.strides
        self.num_channels = self.backbone.num_channels
        self.batch_size = args.batch_size
        # self.edge_model = GIFS_HEATv1(input_dim=128, hidden_dim=256, num_features_levels=4,
        #                               backbone_strides=self.strides,
        #                               backbone_num_channel=self.num_channels)
        # self.corner_model = GIFS_HEATv2(input_dim=128, hidden_dim=256, num_features_levels=4,
        #                                 backbone_strides=self.strides,
        #                                 backbone_num_channel=self.num_channels)
        self.model = GIFS_HEATv3(input_dim=128, hidden_dim=256, num_features_levels=4, backbone_strides=self.strides,
                                 backbone_num_channel=self.num_channels)
        self.device = torch.device("cuda:" + str(args.gpu))
        self.train_dataset = floor_data.FloorPlanDataset(
            "train",
            data_dir=args.data_dir,
            is_raw=args.is_raw,
            rand_aug=args.aug,
        )
        self.val_dataset = floor_data.FloorPlanDataset(
            "val",
            data_dir=args.data_dir,
            is_raw=args.is_raw,
            rand_aug=False
        )
        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.lr = args.lr
        backbone_params = [p for p in self.backbone.parameters()]
        model_params = [p for p in self.model.parameters()]
        all_params = backbone_params + model_params
        self.optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, args.lr_drop)
        self.epochs = args.epochs
        self.alpha = 1

    def train(self):
        wandb.init(project="0316merge")
        self.backbone = nn.DataParallel(self.backbone)
        self.model = nn.DataParallel(self.model)
        # self.backbone = self.backbone.to(self.device)
        # self.edge_model = self.edge_model.to(self.device)
        self.backbone = self.backbone.cuda()
        self.model = self.model.cuda()
        val_min = 1e8
        for epoch in range(self.epochs):
            self.backbone.train()
            self.model.train()
            # self.update_learning_rate(epoch)
            sum_loss = 0
            print('epoch:', epoch)
            for batch in tqdm(self.train_data_loader):
                self.optimizer.zero_grad()
                p = batch.get('grid_coords')
                # inputs = batch.get('inputs').to(self.device)
                inputs = batch.get('inputs').cuda()
                image_feats, feat_mask, all_image_feats = self.backbone(inputs)
                pixels, pixel_features = get_pixel_features(image_size=256)
                # pixel_features = pixel_features.to(self.device)
                pixel_features = pixel_features.cuda()
                pixel_features = pixel_features.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
                gifs_pred, corner_pred = self.train_batch(image_feats, feat_mask, pixel_features, pixels,
                                                          all_image_feats, p)
                # label = batch.get('labels').to(self.device)
                label = batch.get('labels').cuda()
                pixel_labels = batch.get('pixel_labels').cuda()
                # loss = self.edge_loss(gifs_pred, label)
                gifs_loss = torch.nn.BCELoss()(gifs_pred, label)
                corner_loss = torch.nn.BCELoss()(corner_pred.double(), pixel_labels)
                loss = gifs_loss + self.alpha * corner_loss
                # loss = torch.nn.L1Loss()(gifs_pred, label)
                loss.backward()
                self.optimizer.step()
                sum_loss += loss
                wandb.log({"train_loss_last_batch": loss, "epoch": epoch})
            print('sum loss: ', sum_loss)
            wandb.log({"train_loss_avg": sum_loss / len(self.train_data_loader), "epoch": epoch})
            self.backbone.eval()
            self.model.eval()
            sum_val_loss = 0
            for val_batch in self.val_data_loader:
                p = val_batch.get('grid_coords')
                # inputs = val_batch.get('inputs').to(self.device)
                inputs = val_batch.get('inputs').cuda()
                with torch.no_grad():
                    image_feats, feat_mask, all_image_feats = self.backbone(inputs)
                    pixels, pixel_features = get_pixel_features(image_size=256)
                    # pixel_features = pixel_features.to(self.device)
                    pixel_features = pixel_features.cuda()
                    pixel_features = pixel_features.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
                    gifs_pred, corner_pred = self.train_batch(image_feats, feat_mask, pixel_features, pixels,
                                                              all_image_feats, p)
                    # label = val_batch.get('labels').to(self.device)
                    label = val_batch.get('labels').cuda()
                    pixel_labels = val_batch.get('pixel_labels').cuda()
                    # loss = self.edge_loss(gifs_pred, label)
                    gifs_loss = torch.nn.BCELoss()(gifs_pred, label)
                    corner_loss = torch.nn.BCELoss()(corner_pred.double(), pixel_labels)
                    loss = gifs_loss + self.alpha * corner_loss
                    # loss = self.edge_loss(gifs_pred, label)
                sum_val_loss += loss.item()
            print('val_loss:', sum_val_loss)
            wandb.log({"val_loss": sum_val_loss, "epoch": epoch})
            save_path = os.path.join(self.exp_dir, 'last.tar')
            print('save checkpoints: ', save_path)
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_state_dict": self.backbone.state_dict(),
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "args": self.args
                },
                save_path,
            )
            if sum_val_loss < val_min:
                val_min = sum_val_loss
                print('val_min: ', val_min)
                save_path = os.path.join(self.exp_dir, 'best.tar')
                print('save checkpoints: ', save_path)
                torch.save(
                    {
                        "epoch": epoch,
                        "backbone_state_dict": self.backbone.state_dict(),
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "args": self.args
                    },
                    save_path,
                )

    def train_batch(self, image_feats, feat_mask, pixel_features, pixels, all_image_feats, p):

        gifs_pred, corner_pred = self.model(image_feats, feat_mask, pixel_features, p.cuda(), all_image_feats)

        return gifs_pred, corner_pred

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
    parser.add_argument("--data_dir", action="store", dest="data_dir", default='data/20230314_ndc', type=str,
                        help="data dir")
    parser.add_argument("--epochs", action="store", dest="epochs", default=400, type=int, help="epochs")
    parser.add_argument("--batch_size", action="store", dest="batch_size", default=8, type=int, help="batch_size")
    parser.add_argument("--is_raw", action="store", dest="is_raw", default=True, type=bool, help="is_raw")
    parser.add_argument("--aug", action="store", dest="aug", default=True, type=bool, help="aug")
    parser.add_argument('--lr_drop', default=300, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    FLAGS = parser.parse_args()
    print(FLAGS)
    runner = Runner(FLAGS)
    runner.train()
