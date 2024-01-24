import os
import time
from glob import glob

import numpy as np
import torch
import torch.optim as optim
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

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

    point_model = GIFS_HEATv2(input_dim=128, hidden_dim=256)

    device = torch.device(rank)
    point_model = point_model.to(rank)
    # net = net.to(rank)
    point_model = DDP(point_model, device_ids=[rank], find_unused_parameters=True)
    train_dataset = floor_data.vertex_dataset(
            "train",
            data_dir=cfg.data_dir,
            is_raw=cfg.is_raw,
            rand_aug=cfg.aug,
        )

    train_sampler = DistributedSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=train_sampler)

    val_dataset = floor_data.vertex_dataset(
            "val",
            data_dir=cfg.data_dir,
            is_raw=cfg.is_raw,
            rand_aug=False,
        )

    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_data_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, sampler=val_sampler)
    point_params = [p for p in point_model.parameters()]
    optimizer = torch.optim.AdamW(point_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    exp_path = os.path.dirname(__file__) + "/experiments/{}/".format(cfg.exp_name)
    checkpoint_path = exp_path + "checkpoints/".format(cfg.exp_name)
    if not os.path.exists(checkpoint_path):
        if rank == 0:
            print(checkpoint_path)
            os.makedirs(checkpoint_path)
    # if rank == 0:
    #     writer = SummaryWriter(exp_path + 'summary'.format(cfg.exp_name))
    if rank == 0:
        wandb.init(project="0329_point", dir=exp_path, config=cfg, name=cfg.exp_name)

    max_dist = 0.1
    start, training_time = 0, 0

    # ===== load checkpoint =====
    checkpoints = glob(checkpoint_path + "/*")
    if len(checkpoints) == 0:
        if rank == 0:
            print("No checkpoints found at {}".format(checkpoint_path))
    else:
        checkpoints = [os.path.splitext(os.path.basename(path))[0].split("_")[-1] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=float)
        checkpoints = np.sort(checkpoints)
        path = checkpoint_path + "checkpoint_{}h:{}m:{}s_{}.tar".format(
            *[*convertSecs(checkpoints[-1]), checkpoints[-1]]
        )

        if rank == 0:
            print("Loaded checkpoint from: {}".format(path))
        checkpoint = torch.load(path, map_location=f"cuda:{rank}")
        point_model.load_state_dict(checkpoint["point_model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start = checkpoint["epoch"]
        training_time = checkpoint["training_time"]
        val_min = checkpoint["val_min"]

    dist.barrier()

    # ===== train model =====
    loss = 0
    iteration_start_time = time.time()

    for epoch in range(start, cfg.epochs):
        sum_loss = 0
        if rank == 0:
            print("Start epoch {}".format(epoch))

        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        for batch in train_data_loader:
            iteration_duration = time.time() - iteration_start_time

            # ==== save checkpoint and val ====
            save_ckpt_flag = iteration_duration > 60 * 60
            save_ckpt_flag_tensor = torch.tensor(save_ckpt_flag).int().to(device)
            dist.broadcast(save_ckpt_flag_tensor, src=0)
            if save_ckpt_flag_tensor.item() == 1:  # save model every X min and at start
                training_time += iteration_duration
                iteration_start_time = time.time()

                path = checkpoint_path + "checkpoint_{}h:{}m:{}s_{}.tar".format(
                    *[*convertSecs(training_time), training_time]
                )
                if rank == 0:
                    if not os.path.exists(path):
                        torch.save(
                            {  # 'state': torch.cuda.get_rng_state_all(),
                                "training_time": training_time,
                                "epoch": epoch,
                                "point_model_state_dict": point_model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "val_min": val_min,
                            },
                            path,
                        )
                point_model.eval()

                sum_val_loss = 0
                num_batches = 125  # val_data_num = num_batches * batch_size
                val_data_iterator = val_data_loader.__iter__()
                for _ in range(num_batches):
                    try:
                        val_batch = val_data_iterator.next()
                    except:
                        val_data_iterator = val_data_loader.__iter__()
                        val_batch = val_data_iterator.next()

                    inputs = val_batch.get("inputs").to(device)

                    with torch.no_grad():
                        pixel_labels = val_batch.get('pixel_labels').cuda()
                        with torch.cuda.amp.autocast():
                            pixels, pixel_features = get_pixel_features(image_size=256)
                            pixel_features = pixel_features.to(rank)
                            pixel_features = pixel_features.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
                            pred_dict = point_model(inputs, pixels, pixel_features)

                        loss = torch.nn.BCELoss()(pred_dict.double(), pixel_labels)
                    sum_val_loss += loss.item()

                val_loss = sum_val_loss / num_batches

                val_loss_tensor = torch.tensor(val_loss).to(device)
                dist.all_reduce(val_loss_tensor)
                val_loss = val_loss_tensor.item() / world_size

                if val_loss < val_min:
                    val_min = val_loss
                    if rank == 0:
                        for path in glob(exp_path + "val_min=*"):
                            os.remove(path)
                        np.save(
                            exp_path
                            + "val_min={}training_time={}h:{}m:{}s".format(*[epoch, *convertSecs(training_time)]),
                            [epoch, val_loss],
                        )

                if rank == 0:
                    wandb.log({"val_loss": val_loss, "epoch": epoch})

                dist.barrier()

            # ==== optimize model ====
            # backbone.train()
            point_model.train()
            optimizer.zero_grad()

            inputs = batch.get("inputs").to(device)
            label_gt = batch.get("pixel_labels").to(device)

            # image_feats, feat_mask, all_image_feats = backbone(inputs)
            # pixels, pixel_features = get_pixel_features(image_size=256)
            # pixel_features = pixel_features.cuda()
            # pixel_features = pixel_features.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
            # pred_dict = point_model(image_feats, feat_mask, pixel_features, pixels, all_image_feats)
            pixels, pixel_features = get_pixel_features(image_size=256)
            pixel_features = pixel_features.to(rank)
            pixel_features = pixel_features.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
            pred_dict = point_model(inputs, pixels, pixel_features)

            loss = torch.nn.BCELoss()(pred_dict.double(), label_gt)

            loss.backward()
            optimizer.step()

            if rank == 0:
                print("Current loss: {}".format(loss))
            sum_loss += loss

        if rank == 0:
            wandb.log({"train_loss_last_batch": loss, "epoch": epoch})
            wandb.log({"train_loss_avg": sum_loss / len(train_data_loader), "epoch": epoch})

    cleanup()


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", action="store", dest="lr", default=2e-4, type=float, help="Learning rate [0.0001]")
    parser.add_argument("--data_dir", action="store", dest="data_dir", default='data/vertex', type=str,
                        help="data dir")
    parser.add_argument("--epochs", action="store", dest="epochs", default=300, type=int, help="epochs")
    parser.add_argument("--batch_size", action="store", dest="batch_size", default=2, type=int, help="batch_size")
    parser.add_argument("--is_raw", action="store", dest="is_raw", default=True, type=bool, help="is_raw")
    parser.add_argument("--aug", action="store", dest="aug", default=True, type=bool, help="aug")
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    # parser.add_argument('--lr_drop', default=300, type=int)
    parser.add_argument("--exp_name", action="store", dest="exp_name", default='point', type=str)
    cfg = parser.parse_args()
    return cfg


if __name__ == "__main__":
    cfg = get_config()

    n_gpus = torch.cuda.device_count()

    launch(ddp_trainer, cfg, n_gpus)
