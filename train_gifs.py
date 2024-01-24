import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import floor_data
from models.resnet import ResNetBackbone
from models.GIFS_HEAT import GIFS_HEAT, GIFS_HEATv1
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
        self.edge_model = GIFS_HEATv1(input_dim=128, hidden_dim=256, num_features_levels=4,
                                      backbone_strides=self.strides,
                                      backbone_num_channel=self.num_channels)
        self.device = torch.device("cuda:"+str(args.gpu))
        self.train_dataset = floor_data.FloorPlanDataset(
            "train",
            data_dir=args.data_dir,
        )
        self.val_dataset = floor_data.FloorPlanDataset(
            "val",
            data_dir=args.data_dir,
        )
        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.lr = args.lr
        backbone_params = [p for p in self.backbone.parameters()]
        corner_params = [p for p in self.edge_model.parameters()]
        all_params = backbone_params + corner_params
        self.optimizer = optim.Adam(all_params, lr=self.lr)
        self.epochs = args.epochs

    def train(self):
        wandb.init(project="0306gifs")
        self.backbone = self.backbone.to(self.device)
        self.edge_model = self.edge_model.to(self.device)
        val_min = 1e8
        for epoch in range(self.epochs):
            self.backbone.train()
            self.edge_model.train()
            self.update_learning_rate(epoch)
            sum_loss = 0
            print('epoch:', epoch)
            for batch in tqdm(self.train_data_loader):
                self.optimizer.zero_grad()
                p = batch.get('grid_coords')
                inputs = batch.get('inputs').to(self.device)
                label = batch.get('labels').to(self.device)
                image_feats, feat_mask, _ = self.backbone(inputs)
                pixels, pixel_features = get_pixel_features(image_size=256)
                pixel_features = pixel_features.to(self.device)
                pixel_features = pixel_features.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
                gifs_pred = self.train_batch(image_feats, feat_mask, pixel_features, p)
                loss = torch.nn.BCELoss()(gifs_pred, label)
                loss.backward()
                self.optimizer.step()
                sum_loss += loss
                wandb.log({"train_loss_last_batch": loss, "epoch": epoch})
            print('sum loss: ', sum_loss)
            wandb.log({"train_loss_avg": sum_loss / len(self.train_data_loader), "epoch": epoch})
            self.backbone.eval()
            self.edge_model.eval()
            sum_val_loss = 0
            for val_batch in self.val_data_loader:
                p = val_batch.get('grid_coords')
                inputs = val_batch.get('inputs').to(self.device)
                with torch.no_grad():
                    image_feats, feat_mask, _ = self.backbone(inputs)
                    pixels, pixel_features = get_pixel_features(image_size=256)
                    pixel_features = pixel_features.to(self.device)
                    pixel_features = pixel_features.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
                    label = val_batch.get('labels').to(self.device)
                    gifs_pred = self.train_batch(image_feats, feat_mask, pixel_features, p)
                    loss = torch.nn.BCELoss()(gifs_pred, label)
                sum_val_loss += loss.item()
            print('val_loss:', sum_val_loss)
            wandb.log({"val_loss": sum_val_loss, "epoch": epoch})
            save_path = os.path.join(self.exp_dir, 'last.tar')
            print('save checkpoints: ', save_path)
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_state_dict": self.backbone.state_dict(),
                    "point_model_state_dict": self.edge_model.state_dict(),
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
                        "point_model_state_dict": self.edge_model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "args": self.args
                    },
                    save_path,
                )

    def train_batch(self, image_feats, feat_mask, pixel_features, p):
        n = p.shape[1]
        ql = 0
        preds = []
        bsize = 1000
        while ql < n:
            qr = min(ql + bsize, n)
            # print(qr)
            pred = self.edge_model(image_feats, feat_mask, pixel_features, p[:, ql: qr, :].to(self.device))
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
        return pred

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
    parser.add_argument("--lr", action="store", dest="lr", default=0.0001, type=float, help="Learning rate [0.0001]")
    parser.add_argument("--gpu", action="store", dest="gpu", default="0", help="to use which GPU [0]")
    parser.add_argument("--data_dir", action="store", dest="data_dir", default='data/20230306_gifs',type=str, help="data dir")
    parser.add_argument("--epochs", action="store", dest="epochs", default=500, type=int, help="epochs")
    parser.add_argument("--batch_size", action="store", dest="batch_size", default=2, type=int, help="batch_size")
    FLAGS = parser.parse_args()
    print(FLAGS)
    runner = Runner(FLAGS)
    runner.train()
