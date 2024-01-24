import os
import traceback
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import cv2
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from scipy.ndimage import gaussian_filter
import skimage
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import ImageFilter
from PIL import Image


class ImageFilterTransform(object):

    def __init__(self):
        raise NotImplementedError

    def __call__(self, img):
        return img.filter(self.filter)


class GaussianBlur(ImageFilterTransform):
    def __init__(self, radius=2.):
        self.filter = ImageFilter.GaussianBlur(radius=radius)


def random_blur(radius=2.):
    blur = GaussianBlur(radius=radius)
    full_transform = transforms.RandomApply([blur], p=.3)
    return full_transform


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def read_list(filename):
    data_list = []
    file_path = os.path.join('data/s3d_floorplan', filename + '.txt')
    with open(file_path, 'r', encoding='utf-8') as infile:
        for name in infile:
            data_name = name.strip('\n').split()[0]
            data_list.append(data_name)
    return data_list


class FloorPlanDataset(Dataset):
    def __init__(
            self,
            mode,
            data_dir,
            rand_aug=True,
    ):
        self.rand_aug = rand_aug
        self.mode = mode
        if self.mode == 'train':
            self.data_list = read_list('train_list')
        elif self.mode == 'val':
            self.data_list = read_list('valid_list')
        elif self.mode == 'test':
            self.data_list = read_list('test_list')
        self.label_folder = data_dir
        self.grid_list = [32]
        self.image_size = 256
        self.sample_num = 1200

    def __len__(self):
        return len(self.data_list)

    def get_corner_labels(self, corners):
        labels = np.zeros((self.image_size, self.image_size))
        corners = corners.round()
        xint, yint = corners[:, 0].astype(np.int64), corners[:, 1].astype(np.int64)
        labels[yint, xint] = 1

        gauss_labels = gaussian_filter(labels, sigma=2)
        gauss_labels = gauss_labels / gauss_labels.max()
        return labels, gauss_labels

    def __getitem__(self, idx):
        data_name = self.data_list[idx]
        if data_name in ['00404', '00425', '01312', '01424', '02032']:
            return self.__getitem__(np.random.randint(self.__len__()))
        density_path = os.path.join('data/s3d_floorplan', 'density', data_name + '.png')
        img = cv2.imread(density_path)
        label_name = os.path.join(self.label_folder, data_name + '.npz')
        label_dict = np.load(label_name, allow_pickle=True)
        coords = label_dict['coords']
        flags = label_dict['flags']
        flags = np.expand_dims(flags, axis=1)
        vertex_name = os.path.join('./data/coco_vertex_polygon', data_name + '.npz')
        vertex_dict = np.load(vertex_name, allow_pickle=True)
        vertex_coords = vertex_dict['vertex_coords']
        vertex_flags = vertex_dict['vertex_flags']
        polygon_dict = vertex_dict['polygon_dict'].item()
        # vertex_dict = label_dict['vertex']
        # grid_size = 32
        # vertex = []
        # for x in range(grid_size):
        #     for y in range(grid_size):
        #         if vertex_dict[x, y, 0] == -10 or vertex_dict[x, y, 1] == -10:
        #             continue
        #         vertex.append(vertex_dict[x, y, :])
        # vertex = np.array(vertex, dtype=np.float32)
        # coords[coords == 256.0] = 255.0
        if self.rand_aug:
            img, annots = self.random_aug_annot(img, {'coords': coords, 'flags': flags, 'vertex_coords': vertex_coords,
                                                      'vertex_flags': vertex_flags}, is_flip=True, is_rotate=False)
            coords, flags, vertex_coords, vertex_flags = annots['coords'], annots['flags'], annots['vertex_coords'], annots['vertex_flags']
        # vis_img = np.clip(img, 0, 255).astype(np.uint8)
        # for i in range(vertex_flags.shape[0]):
        #     if vertex_flags[i]:
        #         coord = vertex_coords[i, :]
        #         cv2.circle(vis_img, tuple([int(coord[0]), int(coord[1])]), 1, [0, 255, 0], -1)
        # cv2.imshow('1', vis_img)
        # cv2.waitKey(0)
        img = skimage.img_as_float(img)
        img = img.transpose((2, 0, 1))
        # img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[:, np.newaxis, np.newaxis]
        img = img / 255.0
        img = img.astype(np.float32)

        # 分为train 或val test
        # print('train')
        if self.mode == 'train':
            # coords += np.random.normal(0, 0, size=coords.shape)
            vertex_coords += np.random.normal(0, 0, size=vertex_coords.shape)
        # coords += np.random.normal(0, 0, size=coords.shape)
        select_idx = np.random.choice(len(flags), self.sample_num, replace=False)
        coords = coords[select_idx, :]
        flags = flags[select_idx, :]
        assert len(flags) == self.sample_num
        assert len(coords) == self.sample_num
        # coords = coords / 255.0
        vertex_coords = vertex_coords / 255.0
        polygon_num = polygon_dict['polygon_num']
        polygon_len = polygon_dict['polygon_len']
        polygon_coordinate = polygon_dict['polygon_coordinate'] / 255.0
        polygon_index = polygon_dict['polygon_index']
        return {
            'inputs': img,
            'coords': coords,
            'flags': flags,
            'vertex_coords': vertex_coords,
            'vertex_flags': vertex_flags,
            'polygon_num': polygon_num,
            'polygon_len': polygon_len,
            'polygon_coordinate': polygon_coordinate,
            'polygon_index': polygon_index
        }

    def random_flip(self, img, annot):
        height, width, _ = img.shape
        rand_int = np.random.randint(0, 4)
        if rand_int == 0:
            return img, annot
        new_coords = np.array(annot['coords'])

        vertex_coords = np.array(annot['vertex_coords'])
        vertex_flags = np.array(annot['vertex_flags'])
        new_vertex_flags = np.zeros_like(vertex_flags)
        gt_vertex_float = vertex_coords.reshape(32, 32, 2)

        if rand_int == 1:
            img = img[:, ::-1, :]
            new_coords[:, 0] = width - new_coords[:, 0]
            new_coords[:, 2] = width - new_coords[:, 2]
            flip_matrix = np.flip(gt_vertex_float, axis=1)
            for i in range(32):
                for j in range(32):
                    vertex = flip_matrix[i, j, :]
                    if vertex[0] or vertex[1]:
                        flip_matrix[i, j, 0] = width - flip_matrix[i, j, 0]
            flip_matrix = flip_matrix.reshape(32 * 32, 2)
            for i in range(len(flip_matrix)):
                v = flip_matrix[i, :]
                if v[0] or v[1]:
                    new_vertex_flags[i] = 1
            new_vertex_coords = flip_matrix
        elif rand_int == 2:
            img = img[::-1, :, :]
            new_coords[:, 1] = height - new_coords[:, 1]
            new_coords[:, 3] = height - new_coords[:, 3]
            flip_matrix = np.flip(gt_vertex_float, axis=0)
            for i in range(32):
                for j in range(32):
                    vertex = flip_matrix[i, j, :]
                    if vertex[0] or vertex[1]:
                        flip_matrix[i, j, 1] = height - flip_matrix[i, j, 1]
            flip_matrix = flip_matrix.reshape(32 * 32, 2)
            for i in range(len(flip_matrix)):
                v = flip_matrix[i, :]
                if v[0] or v[1]:
                    new_vertex_flags[i] = 1
            new_vertex_coords = flip_matrix
        else:
            img = img[::-1, ::-1, :]
            new_coords[:, 0] = width - new_coords[:, 0]
            new_coords[:, 1] = height - new_coords[:, 1]
            new_coords[:, 2] = width - new_coords[:, 2]
            new_coords[:, 3] = height - new_coords[:, 3]
            flip_matrix = np.flip(gt_vertex_float, axis=1)
            flip_matrix = np.flip(flip_matrix, axis=0)
            for i in range(32):
                for j in range(32):
                    vertex = flip_matrix[i, j, :]
                    if vertex[0] or vertex[1]:
                        flip_matrix[i, j, 0] = width - flip_matrix[i, j, 0]
                        flip_matrix[i, j, 1] = height - flip_matrix[i, j, 1]
            flip_matrix = flip_matrix.reshape(32 * 32, 2)
            for i in range(len(flip_matrix)):
                v = flip_matrix[i, :]
                if v[0] or v[1]:
                    new_vertex_flags[i] = 1
            new_vertex_coords = flip_matrix
        new_coords = np.clip(new_coords, 0, self.image_size - 1)  # clip into [0, 255]
        new_vertex_coords = np.clip(new_vertex_coords, 0, self.image_size - 1)
        aug_annot = dict()
        aug_annot['flags'] = annot['flags']
        aug_annot['coords'] = new_coords
        aug_annot['vertex_coords'] = new_vertex_coords
        aug_annot['vertex_flags'] = new_vertex_flags
        return img, aug_annot

    def random_aug_annot(self, img, annot, is_flip=True, is_rotate=False):
        # do random flipping
        if is_flip and not is_rotate:
            img, annot = self.random_flip(img, annot)
            return img, annot
        if is_rotate:
            # rand_int = np.random.randint(0, 4)
            # if rand_int == 0:
            #     return img, annot
            # prepare random augmentation parameters (only do random rotation for now)
            theta_list = [0, 90, 180, 270]
            theta_idx = theta_list[np.random.randint(len(theta_list))]
            # theta = np.random.randint(0, 360) / 360 * np.pi * 2
            theta = theta_idx / 360 * np.pi * 2
            # theta = np.random.randint(0, 360) / 360 * np.pi * 2
            r = self.image_size / 256
            origin = [127 * r, 127 * r]
            p1_new = [127 * r + 100 * np.sin(theta) * r, 127 * r - 100 * np.cos(theta) * r]
            p2_new = [127 * r + 100 * np.cos(theta) * r, 127 * r + 100 * np.sin(theta) * r]
            p1_old = [127 * r, 127 * r - 100 * r]  # y_axis
            p2_old = [127 * r + 100 * r, 127 * r]  # x_axis
            pts1 = np.array([origin, p1_old, p2_old]).astype(np.float32)
            pts2 = np.array([origin, p1_new, p2_new]).astype(np.float32)
            M_rot = cv2.getAffineTransform(pts1, pts2)
            aug_annot = annot

            new_vertex = np.array(annot['vertex'])
            ones = np.ones([new_vertex.shape[0], 1])
            new_vertex = np.concatenate([new_vertex, ones], axis=-1)
            aug_vertex = np.matmul(M_rot, new_vertex.T).T
            aug_annot['vertex'] = aug_vertex
            if aug_vertex.min() < 0 or aug_vertex.max() > (self.image_size - 1):
                return img, annot

            new_coords = np.array(annot['coords'])
            new_coords_0 = new_coords[:, :2]
            new_coords_1 = new_coords[:, 2:]
            ones = np.ones([new_coords_0.shape[0], 1])
            new_coords_0 = np.concatenate([new_coords_0, ones], axis=-1)
            aug_coords_0 = np.matmul(M_rot, new_coords_0.T).T
            new_coords_1 = np.concatenate([new_coords_1, ones], axis=-1)
            aug_coords_1 = np.matmul(M_rot, new_coords_1.T).T
            aug_coords = np.concatenate([aug_coords_0, aug_coords_1], axis=1)
            aug_coords_list = []
            flags = annot['flags']
            aug_flags = []
            for idx in range(aug_coords.shape[0]):
                # if aug_coords[idx, :].all() == aug_coords_clip[idx, :].all():
                if aug_coords[idx, :].min() >= 0 and aug_coords[idx, :].max() <= 255:
                    aug_coords_list.append(aug_coords[idx, :])
                    aug_flags.append(flags[idx])
            aug_coords = np.array(aug_coords_list, dtype=np.float32)
            aug_flags = np.array(aug_flags, dtype=np.float32)
            aug_annot['flags'] = aug_flags
            aug_annot['coords'] = aug_coords

            rows, cols, ch = img.shape
            new_img = cv2.warpAffine(img, M_rot, (cols, rows), borderValue=(0, 0, 0))
            y_start = (new_img.shape[0] - self.image_size) // 2
            x_start = (new_img.shape[1] - self.image_size) // 2
            aug_img = new_img[y_start:y_start + self.image_size, x_start:x_start + self.image_size, :]

            return aug_img, aug_annot


if __name__ == '__main__':
    dataset = FloorPlanDataset(
        "train",
        data_dir='data/vertex_32',
        rand_aug=False,
    )
    # dataset = vertex_dataset(
    #     "val",
    #     data_dir='data/vertex',
    #     is_raw=True,
    #     rand_aug=True,
    # )
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    for i, data in tqdm(enumerate(data_loader)):
        # print(i, data)
        # poly_gon_index = data['polygon_index']
        # if poly_gon_index.min() < 0 or poly_gon_index.max() > 31:
        #     print(f'error:{i}')
        polygon_len = data['polygon_len']
        filter_data = polygon_len[polygon_len != -1]
        if filter_data.min() <= 0:
            print(f'error:{i}')

        # TODO: 可视化数据增强后的数据是否有问题
        # return {
        #     'inputs': img,
        #     'coords': coords,
        #     'vertex_coords': vertex_coords,
        #     'vertex_flags': vertex_flags,
        #     'vertex_map': vertex_map
        # }
        # img = data['inputs'].squeeze(0).permute(2, 1, 0).detach().cpu().numpy()
        # vertex_coords = data['vertex_coords'].squeeze(0).detach().cpu().numpy()
        # vertex_flags = data['vertex_flags'].sque        #     'flags': flags,eze(0).detach().cpu().numpy()
        # cv2.imshow('1', img)
        # cv2.waitKey(0)

