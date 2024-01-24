import os
import time
from glob import glob
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.optim as optim
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from models.loss import sigmoid_focal_loss
import wandb
import floor_data
from models.resnet import ResNetBackbone
from models.GIFS_HEAT import GIFS_HEAT, GIFS_HEATv1, GIFS_HEATv2
from datasets.data_utils import collate_fn, get_pixel_features
from tqdm import tqdm
import math
import argparse


def convertMillis(millis):
    seconds = int((millis / 1000) % 60)
    minutes = int((millis / (1000 * 60)) % 60)
    hours = int((millis / (1000 * 60 * 60)))
    return hours, minutes, seconds


def convertSecs(sec):
    seconds = int(sec % 60)
    minutes = int((sec / 60) % 60)
    hours = int((sec / (60 * 60)))
    return hours, minutes, seconds


def setup(rank, world_size, port="10239"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def launch(main_fn, cfg, world_size):
    mp.spawn(main_fn, args=(world_size, cfg), nprocs=world_size, join=True)


def ddp_trainer(rank, world_size, cfg):
    setup(rank, world_size)
    edge_model = GIFS_HEATv1(input_dim=128, hidden_dim=256)
    device = torch.device(rank)
    edge_model = edge_model.to(rank)
    # net = net.to(rank)
    edge_model = DDP(edge_model, device_ids=[rank], find_unused_parameters=True)
    train_dataset = floor_data.FloorPlanDataset(
        "train",
        data_dir=cfg.data_dir,
        is_raw=cfg.is_raw,
        rand_aug=cfg.aug,
    )
    train_sampler = DistributedSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=train_sampler, num_workers=4)

    val_dataset = floor_data.FloorPlanDataset(
        "val",
        data_dir=cfg.data_dir,
        is_raw=cfg.is_raw,
        rand_aug=False,
    )
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_data_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler)
    params = [p for p in edge_model.parameters()]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.lr_drop)
    timestamp = time.strftime('%Y%m%d_%H_%M_%S', time.localtime())
    exp_dir = os.path.join('experiments', timestamp)

    val_min = 1e8
    f1_max = -1
    recall_max = -1
    # if rank == 0:
    #     writer = SummaryWriter(exp_path + 'summary'.format(cfg.exp_name))
    if not os.path.exists(exp_dir):
        if rank == 0:
            print(exp_dir)
            os.makedirs(exp_dir)
    if rank == 0:
        wandb.init(project="0404_gifs", dir=exp_dir, config=cfg, name=cfg.exp_name)

    # ===== train model =====
    loss = 0
    start = 0
    for epoch in range(start, cfg.epochs):
        sum_loss = 0
        sum_hb = 0
        sum_rel = 0
        if rank == 0:
            print("Start epoch {}".format(epoch))
        iteration_start_time = time.time()
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        for batch in train_data_loader:
            # ==== optimize model ====
            # backbone.train()
            edge_model.train()
            optimizer.zero_grad()

            inputs = batch.get("inputs").to(device)
            p = batch.get('coords').to(device)
            label = batch.get("flags").to(device)
            # mask = batch.get('mask').to(device)
            # mask = batch.get("mask").to(device)
            pixels, pixel_features = get_pixel_features(image_size=256)
            pixel_features = pixel_features.to(device)
            pixel_features = pixel_features.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
            gifs_hb = edge_model(inputs, pixel_features, p)
            # hb_loss = sigmoid_focal_loss(gifs_hb, label, alpha=0.8, reduction='sum')
            # rel_loss = sigmoid_focal_loss(gifs_rel, label, alpha=0.8, reduction='sum')
            # hb_loss = torch.nn.BCELoss()(gifs_hb, label)
            # rel_loss = torch.nn.BCELoss()(gifs_rel, label)
            # true_num, false_num = batch.get("true_num").detach().numpy(), batch.get("false_num").detach().numpy()
            # true_avg_num, false_avg_num = true_num.mean(), false_num.mean()
            # true_factor, false_factor = false_avg_num / true_avg_num, false_avg_num / false_avg_num
            # hb_loss = nn.CrossEntropyLoss(
            #     weight=torch.tensor([false_factor.astype('float32'), true_factor.astype('float32')]).to(device),
            #     reduction='mean')(gifs_hb.permute(0, 2, 1), label)
            # rel_loss = nn.CrossEntropyLoss(
            #     weight=torch.tensor([1.0, 2.0]).to(device),
            #     reduction='mean')(gifs_rel.permute(0, 2, 1), label)
            # hb_loss = edge_loss(gifs_hb[mask], label[mask].squeeze(1))
            # hb_loss = edge_loss(gifs_hb, label.squeeze(2))
            hb_loss = F.binary_cross_entropy_with_logits(gifs_hb, label)
            # rel_loss = edge_loss(gifs_rel.permute(0, 2, 1), label)
            # loss = hb_loss + rel_loss
            loss = hb_loss
            loss.backward()
            optimizer.step()

            if rank == 0:
                print("Current loss: {}".format(loss))
                wandb.log({"train_loss_last_batch": loss, "epoch": epoch})
            sum_loss += loss
            sum_hb += hb_loss
            # sum_rel += rel_loss
        lr_scheduler.step()

        if rank == 0:
            wandb.log({"train_loss_avg": sum_loss / len(train_data_loader), "epoch": epoch})
            # wandb.log({"hb_loss_avg": sum_hb / len(train_data_loader), "epoch": epoch})
            # wandb.log({"rel_loss_avg": sum_rel / len(train_data_loader), "epoch": epoch})
            training_time = time.time() - iteration_start_time
            print('time: ', training_time)
            print('train_loss: ', sum_loss / len(train_data_loader))

        # ==== save checkpoint and val ====
        save_ckpt_flag = 1
        save_ckpt_flag_tensor = torch.tensor(save_ckpt_flag).int().to(device)
        dist.broadcast(save_ckpt_flag_tensor, src=0)
        if save_ckpt_flag_tensor.item() == 1:  # save model every X min and at start
            path = os.path.join(exp_dir, 'last.tar')
            if rank == 0:
                torch.save(
                    {  # 'state': torch.cuda.get_rng_state_all(),
                        "epoch": epoch,
                        "edge_model_state_dict": edge_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    path,
                )
            edge_model.eval()
            sum_f1 = 0
            for val_batch in val_data_loader:
                inputs = val_batch.get("inputs").to(device)
                p = val_batch.get('coords').to(device)
                with torch.no_grad():
                    pixels, pixel_features = get_pixel_features(image_size=256)
                    pixel_features = pixel_features.to(device)
                    pixel_features = pixel_features.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
                    gifs_hb = edge_model(inputs, pixel_features, p)
                    # gifs_pred = torch.sigmoid(gifs_hb) > 0.5
                pred_gt = (torch.sigmoid(gifs_hb) > 0.5).squeeze().detach().cpu().numpy().astype(np.float32)
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
                # compute acc, precision and recall

            f1 = sum_f1 / len(val_data_loader)
            f1_tensor = torch.tensor(f1).to(device)
            dist.all_reduce(f1_tensor)
            f1 = f1_tensor.item() / world_size

            # pred = all_sum_correct / all_sum_prec
            # recall = all_sum_correct / all_sum_recall
            #
            # pred_tensor = torch.tensor(pred).to(device)
            # dist.all_reduce(pred_tensor)
            # pred = pred_tensor.item() / world_size
            #
            # recall_tensor = torch.tensor(recall).to(device)
            # dist.all_reduce(recall_tensor)
            # recall = recall_tensor.item() / world_size

            if f1 >= f1_max:
                f1_max = f1
                # recall_max = recall
                path = os.path.join(exp_dir, 'best.tar')
                if rank == 0:
                    torch.save(
                        {  # 'state': torch.cuda.get_rng_state_all(),
                            "epoch": epoch,
                            "edge_model_state_dict": edge_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        path,
                    )
            if rank == 0:
                print('acc:', f1)
                # print("pred", pred)
                # print("recall", recall)
                wandb.log({"acc": f1, "epoch": epoch})
                # wandb.log({"pred": pred, "epoch": epoch})
                # wandb.log({"recall": recall, "epoch": epoch})

            dist.barrier()

    cleanup()


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", action="store", dest="lr", default=1e-4, type=float, help="Learning rate [0.0001]")
    parser.add_argument("--data_dir", action="store", dest="data_dir", default='data/20230410', type=str,
                        help="data dir")
    parser.add_argument("--epochs", action="store", dest="epochs", default=500, type=int, help="epochs")
    parser.add_argument("--batch_size", action="store", dest="batch_size", default=4, type=int, help="batch_size")
    parser.add_argument("--is_raw", action="store", dest="is_raw", default=True, type=bool, help="is_raw")
    parser.add_argument("--aug", action="store", dest="aug", default=True, type=bool, help="aug")
    parser.add_argument('--lr_drop', default=350, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    # parser.add_argument('--lr_drop', default=300, type=int)
    parser.add_argument("--exp_name", action="store", dest="exp_name", default='edge', type=str)
    parser.add_argument("--exp_dir", action="store", dest="exp_dir",
                        default=time.strftime('%Y%m%d_%H_%M_%S', time.localtime()), type=str)
    cfg = parser.parse_args()

    return cfg


if __name__ == "__main__":
    cfg = get_config()

    n_gpus = torch.cuda.device_count()

    launch(ddp_trainer, cfg, n_gpus)
