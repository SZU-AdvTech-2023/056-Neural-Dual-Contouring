import numpy as np
import os
import cv2
import torch
import time


class Generate_Label:
    def __init__(self):
        print('1')
        self.sample_num = 1000
        self.annotation_path = 'data/s3d_floorplan/annot'
        self.density_path = 'data/s3d_floorplan/density'
        timestamp = time.strftime('%Y%m%d', time.localtime())
        self.label_folder = os.path.join('data', timestamp + '_gifs')
        # 用于training 的点是基于groundTruth的，用于validation和test的点是随机采样的
        if not os.path.exists(self.label_folder):
            os.mkdir(self.label_folder)
        self.train_file_list = self.read_list('train_list')
        self.valid_file_list = self.read_list('valid_list')
        self.test_file_list = self.read_list('test_list')
        # self.surface_sample_scales = [2, 4, 8, 16, 32]
        # self.surface_sample_ratios = [0.3, 0.3, 0.3, 0.05, 0.05]  # sum: 0.9
        self.surface_sample_scales = [0.01 * 256, 0.02 * 256, 0.04 * 256]
        self.surface_sample_ratios = [0.5, 0.3, 0.2]  # sum: 0.9
        self.offset = 1e-5
        self.grid_size_list = [16, 32, 64]
        self.train_random_ratio = 0.6
        self.w = 256
        self.gpu = 1

    def generate_udf_label(self, is_training):
        samples = np.ones([256 * 256, 2], dtype=np.float64)
        idx = 0
        for i in range(256):
            for j in range(256):
                samples[idx, :] = [i, j]
                idx += 1

        if is_training == 0:
            file_list = self.train_file_list
            print('Process training set')
        elif is_training == 1:
            file_list = self.valid_file_list
            print('Process val set')
        elif is_training == 2:
            file_list = self.test_file_list
            print('Process test set')
        len_of_point = []
        for file_name in file_list:
            udf = np.zeros([self.sample_num, 1], dtype=np.float64)
            start = time.time()
            print(file_name)
            annot_path = os.path.join(self.annotation_path, file_name + '.npy')
            # img_path = os.path.join(self.density_path, file_name + '.png')
            # img = cv2.imread(img_path)
            annot = np.load(annot_path, allow_pickle=True, encoding='latin1').tolist()
            gt_data = self.convert_annot(annot)
            edges = self.draw_edges(gt_data)
            line_points_list = []
            for edge in edges:
                point0, point1 = edge[0], edge[1]
                lines_points = self.sample_points_on_line_segment(point0, point1, distance=0.1)
                line_points_list.extend(lines_points)
            line_points_list = np.array(line_points_list)
            sample = np.zeros([self.sample_num, 2], dtype=np.float64)
            if len(line_points_list) > 0:
                max_line_num = int(self.sample_num / 2)
                line_sample_num = min(max_line_num, len(line_points_list))
                random_sample_num = self.sample_num - line_sample_num
                select_line_idx = np.random.choice(len(line_points_list), line_sample_num, replace=False)
                sample[:line_sample_num, :] = line_points_list[select_line_idx, :]
                random_sample_idx = np.random.choice(256 * 256, int(random_sample_num), replace=False)
                sample[line_sample_num:, :] = samples[random_sample_idx, :]
                n = self.sample_num
                ql = 0
                bsize = 5000
                while ql < n:
                    qr = min(ql + bsize, n)
                    samples_tmp = sample[ql:qr, :]
                    nearest_idx = self.search_nearest_point(torch.tensor(samples_tmp).float().cuda(self.gpu),
                                                            torch.tensor(line_points_list).float().cuda(self.gpu))
                    sample_gt = line_points_list[nearest_idx, :]
                    udf[ql:qr, :] = self.compute_udf(torch.tensor(samples_tmp).float().cuda(self.gpu),
                                                     torch.tensor(sample_gt).float().cuda(self.gpu))
                    ql = qr
            else:
                random_sample_idx = np.random.choice(256 * 256, int(self.sample_num), replace=False)
                sample = samples[random_sample_idx, :]
            end = time.time()
            print('time: ', end - start)
            # for i in range(self.sample_num):
            #     cv2.circle(img, tuple([int(sample[i, 0]), int(sample[i, 1])]), 1, [0, 0, 255], -1)
            # cv2.imshow('1', img)
            # cv2.waitKey(0)
            # convert sample to  -1 - 1
            # transform_sample = self.transform_sample_coord(sample)
            # sample_near = self.transform_sample_coord(sample_near)
            assert len(sample) == self.sample_num
            assert len(udf) == self.sample_num
            transform_sample = sample
            udf = udf / 255.0
            label_name = file_name + '.npz'
            label_path = os.path.join(self.label_folder, label_name)
            # np.savez(label_path, grid_coords=transform_sample, df=udf, sample_near=sample_near)
            np.savez(label_path, grid_coords=transform_sample, df=udf)

    def generate_label(self):
        self.handle_train_list()
        self.handle_random_list(True)
        self.handle_random_list(False)
        # self.handle_test_list()

    def convert_annot(self, annot):
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

    def draw_edges(self, gt_data):
        edges = gt_data['edges']
        corners = gt_data['corners']
        edge_result = []
        for edge in edges:
            point0 = corners[edge[0], :]
            point1 = corners[edge[1], :]
            if point0[0] == point1[0] and point0[1] == point1[1]:
                continue
            edge_result.append([point0, point1])
            # img = cv2.line(img, tuple(point0), tuple(point1), [255, 255, 255])
        return edge_result

    def compute_distance_point2line(self, point, line):
        '''
        :param point:[x0, y0]
        :param line: [point0, point1]
        :return: distance
        '''
        line_pt0, line_pt1 = np.array(line[0]), np.array(line[1])
        vec1 = line_pt0 - point
        vec2 = line_pt1 - point
        if np.linalg.norm(line_pt0 - line_pt1) < 1e-5:
            print('no')
        distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_pt0 - line_pt1)
        return distance

    def compute_intersect(self, l1, l2):
        '''
        :param l1: [x1 ,y1, x2, y2]
        :param l2: [x1, y1, x2, y2]
        :return:
        '''
        v1 = (l1[0] - l2[0], l1[1] - l2[1])
        v2 = (l1[0] - l2[2], l1[1] - l2[3])
        v0 = (l1[0] - l1[2], l1[1] - l1[3])
        a = (v0[0] * v1[1] - v0[1] * v1[0])
        b = (v0[0] * v2[1] - v0[1] * v2[0])

        temp = l1
        l1 = l2
        l2 = temp
        v1 = (l1[0] - l2[0], l1[1] - l2[1])
        v2 = (l1[0] - l2[2], l1[1] - l2[3])
        v0 = (l1[0] - l1[2], l1[1] - l1[3])
        c = (v0[0] * v1[1] - v0[1] * v1[0])
        d = (v0[0] * v2[1] - v0[1] * v2[0])

        # if a * b == 0 and c * d == 0:
        #     print(1)
        # print('a*b: ', a*b)
        # print('c*d: ', c*d)
        if a * b <= 0 and c * d <= 0 and not (a * b == 0 and c * d == 0):
            return True
        else:
            return False

    def line_general_equation(self, line):
        """直线一般式"""
        A = line[3] - line[1]
        B = line[0] - line[2]
        C = line[2] * line[1] - line[0] * line[3]
        line = np.array([A, B, C])
        if B != 0:
            line = line / B
        return line

    def sample_points_on_line_segment(self, point1, point2, distance=3):
        """在一条线段上均匀采样"""
        line_dist = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)  # 线段长度
        num = round(line_dist / distance)  # 采样段数量
        line = [point1[0], point1[1], point2[0], point2[1]]  # 两点成线
        line_ABC = self.line_general_equation(line)  # 一般式规范化
        newP = []
        newP.append(point1.tolist())  # 压入首端点
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
        newP.append([point2[0], point2[1]])  # 压入末端点
        return newP

    def compute_udf(self, sample, sample_near):
        # if len(sample_near > 0):
        #     distance_1 = torch.sqrt(torch.sum((sample[:, 0:2] - sample_near[:, 0:2]) ** 2, axis=-1)).unsqueeze(1)
        #     distance_2 = torch.sqrt(torch.sum((sample[:, 2:4] - sample_near[:, 2:4]) ** 2, axis=-1)).unsqueeze(1)
        #     distance_1 = np.array(distance_1.cpu())
        #     distance_2 = np.array(distance_2.cpu())
        #     return np.concatenate((distance_1, distance_2), axis=1)
        # else:
        #     return np.zeros((sample.shape[0], 1))
        if len(sample_near > 0):
            distance_1 = torch.sqrt(torch.sum((sample[:, 0:2] - sample_near[:, 0:2]) ** 2, axis=-1)).unsqueeze(1)
            # distance_2 = torch.sqrt(torch.sum((sample[:, 2:4] - sample_near[:, 2:4]) ** 2, axis=-1)).unsqueeze(1)
            distance_1 = np.array(distance_1.cpu())
            # distance_2 = np.array(distance_2.cpu())
            return distance_1
        else:
            return np.zeros((sample.shape[0], 1))

    def compute_flag(self, random_pairs, edges):
        binary_flags = np.zeros((random_pairs.shape[0], 1))
        for idx, random_pair in enumerate(random_pairs):
            binary_flag = False
            for edge in edges:
                edge_line = []
                edge_line.extend(edge[0].tolist())
                edge_line.extend(edge[1].tolist())
                flag = self.compute_intersect(random_pair, edge_line)
                if flag:
                    binary_flag = True
                    break
            binary_flags[idx] = binary_flag
        return binary_flags

    def search_nearest_point(self, point_batch, point_gt):
        num_point_batch, num_point_gt = point_batch.shape[0], point_gt.shape[0]
        point_batch = point_batch.unsqueeze(1).repeat(1, num_point_gt, 1)
        point_gt = point_gt.unsqueeze(0).repeat(num_point_batch, 1, 1)

        distances = torch.sqrt(torch.sum((point_batch - point_gt) ** 2, axis=-1) + 1e-12)
        dis_idx = torch.argmin(distances, axis=1).detach().cpu().numpy()

        return dis_idx

    def read_list(self, filename):
        data_list = []
        file_path = os.path.join('data/s3d_floorplan', filename + '.txt')
        with open(file_path, 'r', encoding='utf-8') as infile:
            for name in infile:
                data_name = name.strip('\n').split()[0]
                data_list.append(data_name)
        return data_list

    def handle_random_list(self, is_valid):
        if is_valid:
            file_list = self.valid_file_list
        else:
            file_list = self.test_file_list
        for file_name in file_list:
            print(file_name)
            start = time.time()
            # print(file_name)
            annot_path = os.path.join(self.annotation_path, file_name + '.npy')
            img_path = os.path.join(self.density_path, file_name + '.png')
            annot = np.load(annot_path, allow_pickle=True, encoding='latin1').tolist()
            img = cv2.imread(img_path)
            gt_data = self.convert_annot(annot)
            edges = self.draw_edges(gt_data)
            line_points_list = []
            for edge in edges:
                point0, point1 = edge[0], edge[1]
                lines_points = self.sample_points_on_line_segment(point0, point1, distance=0.1)
                line_points_list.extend(lines_points)
            line_points_list = np.array(line_points_list)
            sample_point = []
            sample_point_idx = np.random.choice(256 * 256, int(self.sample_num), replace=False)
            for idx in range(len(sample_point_idx)):
                point_idx = sample_point_idx[idx]
                pt_x, pt_y = int(point_idx / 256), point_idx % 256
                sample_point.append([pt_x, pt_y])
                # sample = np.vstack((sample, np.array([pt_x, pt_y], dtype=np.float64)))
            # print(sample_point)
            sample_point = np.array(sample_point, dtype=np.float64)
            # surface_sample_scales = [2, 4, 8, 16, 32, 64]
            # surface_sample_ratios = [0.25, 0.25, 0.25, 0.125, 0.0625, 0.0625]  # sum: 0.9
            sample = []
            sample_near = []
            for sample_ratio, sample_scale in zip(self.surface_sample_ratios, self.surface_sample_scales):
                sample_points_num = int(self.sample_num * sample_ratio)
                select_indices = np.random.choice(len(sample_point), sample_points_num)
                select_list = sample_point[select_indices]
                random_pairs = np.tile(select_list, (1, 2))
                assert random_pairs.shape[1] == 4  # shape: N x 4
                random_pairs = random_pairs + np.random.randn(*random_pairs.shape) * sample_scale
                random_pairs[random_pairs < 0] = 0
                random_pairs[random_pairs > 255] = 255
                first_sample, second_sample = random_pairs[:, 0:2], random_pairs[:, 2:4]
                first_nearest_idx = self.search_nearest_point(torch.tensor(first_sample).float().cuda(self.gpu),
                                                              torch.tensor(line_points_list).float().cuda(self.gpu))
                second_nearest_idx = self.search_nearest_point(torch.tensor(second_sample).float().cuda(self.gpu),
                                                               torch.tensor(line_points_list).float().cuda(self.gpu))
                first_sample_gt = line_points_list[first_nearest_idx, :]
                second_sample_gt = line_points_list[second_nearest_idx, :]
                sample_points = np.concatenate((first_sample, second_sample), axis=1)
                sample_gt = np.concatenate((first_sample_gt, second_sample_gt), axis=1)
                sample.append(sample_points)
                sample_near.append(sample_gt)
            sample = np.concatenate(sample, axis=0)
            sample_near = np.concatenate(sample_near, axis=0)
            udf = self.compute_udf(torch.tensor(sample).float().cuda(self.gpu),
                                   torch.tensor(sample_near).float().cuda(self.gpu))
            binary_flags = self.compute_flag(sample, edges)
            end = time.time()
            print('time: ', end - start)
            # for idx in range(sample.shape[0]):
            #     # if idx >= self.sample_num * 0.9:
            #     print(idx)
            #     # cv2.circle(img, tuple([int(sample[idx, 0]), int(sample[idx, 1])]), 1, [255, 255, 255], -1)
            #     # cv2.circle(img, tuple([int(sample[idx, 2]), int(sample[idx, 3])]), 1, [0, 255, 0], -1)
            #     if binary_flags[idx]:
            #         # print('yes')
            #         cv2.line(img, tuple([int(sample[idx, 0]), int(sample[idx, 1])]),
            #                  tuple([int(sample[idx, 2]), int(sample[idx, 3])]), [0, 0, 255], 1)
            #     else:
            #         cv2.line(img, tuple([int(sample[idx, 0]), int(sample[idx, 1])]),
            #                  tuple([int(sample[idx, 2]), int(sample[idx, 3])]), [0, 255, 0], 1)
            #     # cv2.circle(img, tuple([int(sample_near[idx, 0]), int(sample_near[idx, 1])]), 1,
            #     #            [0, 0, 255], -1)
            #     # print(idx)
            #     # print(udf[idx])
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            assert len(sample) == self.sample_num
            assert len(binary_flags) == self.sample_num
            assert len(udf) == self.sample_num
            # sample = self.transform_sample_coord(sample)
            label_name = file_name + '.npz'
            label_path = os.path.join(self.label_folder, label_name)
            np.savez(label_path, grid_coords=sample, labels=binary_flags, df=udf)

    def convert_grid(self, is_training):
        file_list = []
        if is_training == 0:
            file_list = self.train_file_list
            print('Process training set')
        elif is_training == 1:
            file_list = self.valid_file_list
            print('Process val set')
        elif is_training == 2:
            file_list = self.test_file_list
            print('Process test set')
        for file_name in file_list:
            print(file_name)
            start = time.time()
            gt = dict()
            for grid_size in self.grid_size_list:
                print(grid_size)
                gt['grid_size_' + str(grid_size)] = dict()
                grid_size_1 = grid_size + 1
                size_of_grid = int(256 / grid_size)
                annot_path = os.path.join(self.annotation_path, file_name + '.npy')
                annot = np.load(annot_path, allow_pickle=True, encoding='latin1').tolist()
                gt_data = self.convert_annot(annot)
                edges = self.draw_edges(gt_data)
                img_path = os.path.join(self.density_path, file_name + '.png')
                img = cv2.imread(img_path)
                # for edge in edges:
                #     cv2.line(img, tuple(edge[0]), tuple(edge[1]), [255, 0, 0], 1)
                # cv2.imshow('1', img)
                # cv2.waitKey(0)
                edge_x = np.zeros([grid_size, grid_size_1, 4], dtype=np.float64)
                edge_y = np.zeros([grid_size_1, grid_size, 4], dtype=np.float64)
                gt_edge_x = np.zeros([grid_size, grid_size_1], dtype=np.float64)
                gt_edge_y = np.zeros([grid_size_1, grid_size], dtype=np.float64)
                intersection_x = np.zeros([grid_size, grid_size_1, 2], dtype=np.float64)
                intersection_y = np.zeros([grid_size_1, grid_size, 2], dtype=np.float64)
                intersection_x_normal = np.zeros([grid_size, grid_size_1, 2], dtype=np.float64)
                intersection_y_normal = np.zeros([grid_size_1, grid_size, 2], dtype=np.float64)
                for i in range(grid_size):
                    for j in range(grid_size_1):
                        edge_x[i, j, :] = [i * size_of_grid, j * size_of_grid,
                                           (i + 1) * size_of_grid, j * size_of_grid]
                        binary_flag = False
                        for edge in edges:
                            edge_line = []
                            edge_line.extend(edge[0].tolist())
                            edge_line.extend(edge[1].tolist())
                            edge_x_tmp = edge_x[i, j, :] + self.offset
                            flag = self.compute_intersect(edge_x_tmp, edge_line)
                            if flag:
                                binary_flag = True
                                line1 = LineString(edge)
                                line2 = LineString([[i * size_of_grid, j * size_of_grid],
                                                    [(i + 1) * size_of_grid, j * size_of_grid]])
                                pt = line1.intersection(line2)
                                intersection_point = pt.x, pt.y
                                tangent = edge[1] - edge[0]
                                tangent = tangent / np.linalg.norm(tangent, 2)
                                edge_normal = [-1 * tangent[1], tangent[0]]
                                intersection_x[i, j, :] = [pt.x, pt.y]
                                intersection_x_normal[i, j, :] = edge_normal
                                # print(edge_normal)
                                # print(tangent)
                                break
                        if binary_flag:
                            # print(intersection_point)
                            gt_edge_x[i, j] = 1
                            cv2.line(img, tuple([i * size_of_grid, j * size_of_grid]),
                                 tuple([i * size_of_grid + size_of_grid, j * size_of_grid]), [0, 0, 255], 1)
                            # cv2.circle(img, tuple([int(pt.x), int(pt.y)]), 1, [0, 255, 0], -1)
                        # else:
                        #     cv2.line(img, tuple([i * size_of_grid, j * size_of_grid]),
                        #              tuple([i * size_of_grid + size_of_grid, j * size_of_grid]), [0, 255, 0], 1)
                            # cv2.imshow('1', img)
                            # cv2.waitKey(0)
                            # cv2.imshow('1', img)
                            # cv2.waitKey(0)
                for i in range(grid_size_1):
                    for j in range(grid_size):
                        edge_y[i, j, :] = [i * size_of_grid, j * size_of_grid, i * size_of_grid, (j + 1) * size_of_grid]
                        binary_flag = False
                        for edge in edges:
                            edge_line = []
                            edge_line.extend(edge[0].tolist())
                            edge_line.extend(edge[1].tolist())
                            edge_y_tmp = edge_y[i, j, :] + self.offset
                            flag = self.compute_intersect(edge_y_tmp, edge_line)
                            if flag:
                                binary_flag = True
                                line1 = LineString(edge)
                                line2 = LineString(
                                    [[i * size_of_grid, j * size_of_grid], [i * size_of_grid, (j + 1) * size_of_grid]])
                                pt = line1.intersection(line2)
                                intersection_point = pt.x, pt.y
                                tangent = edge[1] - edge[0]
                                tangent = tangent / np.linalg.norm(tangent, 2)
                                edge_normal = [-1 * tangent[1], tangent[0]]
                                # print(edge_normal)
                                intersection_y[i, j, :] = [pt.x, pt.y]
                                intersection_y_normal[i, j, :] = edge_normal
                                break
                        if binary_flag:
                            # print(intersection_point)
                            gt_edge_y[i, j] = 1
                            cv2.line(img, tuple([i * size_of_grid, j * size_of_grid]),
                                 tuple([i * size_of_grid, (j + 1) * size_of_grid]), [0, 0, 255], 1)
                            # cv2.circle(img, tuple([int(pt.x), int(pt.y)]), 1, [0, 255, 0], -1)
                        # else:
                        #     cv2.line(img, tuple([i * size_of_grid, j * size_of_grid]),
                        #              tuple([i * size_of_grid, (j + 1) * size_of_grid]), [0, 255, 0], 1)
                        #     cv2.imshow('1', img)
                        #     cv2.waitKey(0)
                # cv2.imshow('1', img)
                # cv2.waitKey(0)
                # print(file_name)
                # gt_vertex_float = np.zeros([grid_size, grid_size, 2], dtype=np.float64)
                # for x in range(grid_size):
                #     for y in range(grid_size):
                #         if gt_edge_x[x][y] or gt_edge_x[x][y+1] or gt_edge_y[x][y] or gt_edge_y[x+1][y]:
                #             # print('generate vertex')
                #             intersection_points = []
                #             normals = []
                #             if gt_edge_x[x][y]:
                #                 intersection_points.append(intersection_x[x, y, :])
                #                 normals.append(intersection_x_normal[x, y, :])
                #             if gt_edge_x[x][y+1]:
                #                 intersection_points.append(intersection_x[x, y+1, :])
                #                 normals.append(intersection_x_normal[x, y+1, :])
                #             if gt_edge_y[x][y]:
                #                 intersection_points.append(intersection_y[x, y, :])
                #                 normals.append(intersection_y_normal[x, y, :])
                #             if gt_edge_y[x+1][y]:
                #                 intersection_points.append(intersection_y[x+1, y, :])
                #                 normals.append(intersection_y_normal[x+1, y, :])
                #             # print(intersection_points)
                #             # print(normals)
                #             center_x = size_of_grid / 2 + size_of_grid * x
                #             center_y = size_of_grid / 2 + size_of_grid * y
                #             # v = solve_qef_2d(center_x, center_y, intersection_points, normals)
                #             gt_vertex_float[x, y, :] = [v.x, v.y]
                #             # print(v)
                #             cv2.circle(img, tuple([int(v.x), int(v.y)]), 1, [0, 255, 0], -1)
                # gt['grid_size_'+str(grid_size)]['gt_vertex_float'] = gt_vertex_float
                # gt['grid_size_' + str(grid_size)]['gt_edge_x'] = gt_edge_x
                # gt['grid_size_' + str(grid_size)]['gt_edge_y'] = gt_edge_y
            end = time.time()
            print('time: ', end - start)
            label_name = file_name + '.npz'
            label_path = os.path.join(self.label_folder, label_name)
            # np.savez(label_path, gt=gt)

    def handle_train_list(self):
        for file_name in self.train_file_list:
            start = time.time()
            print(file_name)
            annot_path = os.path.join(self.annotation_path, file_name + '.npy')
            img_path = os.path.join(self.density_path, file_name + '.png')
            annot = np.load(annot_path, allow_pickle=True, encoding='latin1').tolist()
            img = cv2.imread(img_path)
            gt_data = self.convert_annot(annot)
            edges = self.draw_edges(gt_data)
            line_points_list = []
            for edge in edges:
                point0, point1 = edge[0], edge[1]
                lines_points = self.sample_points_on_line_segment(point0, point1, distance=0.1)
                line_points_list.extend(lines_points)
                # cv2.line(img, tuple(point0.astype(int)), tuple(point1.astype(int)), [255, 255, 255])
            line_points_list = np.array(line_points_list)
            if len(line_points_list) > 0:
                line_sample_num = int(self.sample_num * (1 - self.train_random_ratio))
                random_sample_num = int(self.sample_num * self.train_random_ratio)
                # continue
                sample_idx = np.random.choice(len(line_points_list), line_sample_num)
                sample_points_list = line_points_list[sample_idx, :]
                sample = []
                sample_near = []
                for sample_ratio, sample_scale in zip(self.surface_sample_ratios, self.surface_sample_scales):
                    sample_points_num = int(line_sample_num * sample_ratio)
                    select_indices = np.random.choice(len(sample_points_list), sample_points_num)
                    select_list = sample_points_list[select_indices]
                    random_pairs = np.tile(select_list, (1, 2))
                    assert random_pairs.shape[1] == 4  # shape: N x 4
                    random_pairs = random_pairs + np.random.randn(*random_pairs.shape) * sample_scale
                    random_pairs[random_pairs < 0] = 0
                    random_pairs[random_pairs > 256] = 255
                    first_sample, second_sample = random_pairs[:, 0:2], random_pairs[:, 2:4]
                    first_nearest_idx = self.search_nearest_point(torch.tensor(first_sample).float().cuda(self.gpu),
                                                                  torch.tensor(line_points_list).float().cuda(self.gpu))
                    second_nearest_idx = self.search_nearest_point(torch.tensor(second_sample).float().cuda(self.gpu),
                                                                   torch.tensor(line_points_list).float().cuda(
                                                                       self.gpu))
                    first_sample_gt = line_points_list[first_nearest_idx, :]
                    second_sample_gt = line_points_list[second_nearest_idx, :]
                    sample_points = np.concatenate((first_sample, second_sample), axis=1)
                    sample_gt = np.concatenate((first_sample_gt, second_sample_gt), axis=1)
                    sample.append(sample_points)
                    sample_near.append(sample_gt)

                if random_sample_num > 0:
                    # generate_random_pairs
                    sample_point_idx = np.random.choice(256 * 256, random_sample_num, replace=False)
                    random_points = np.zeros([random_sample_num, 2], dtype=np.float64)
                    for idx in range(len(sample_point_idx)):
                        point_idx = sample_point_idx[idx]
                        random_points[idx, :] = [int(point_idx / 256), point_idx % 256]

                    random_pairs = np.tile(random_points, (1, 2))
                    assert random_pairs.shape[1] == 4  # shape: N x 4
                    random_pairs = random_pairs + np.random.randn(*random_pairs.shape) * 5
                    random_pairs[random_pairs < 0] = 0
                    random_pairs[random_pairs > 255] = 255
                    first_sample, second_sample = random_pairs[:, 0:2], random_pairs[:, 2:4]
                    first_nearest_idx = self.search_nearest_point(torch.tensor(first_sample).float().cuda(self.gpu),
                                                                  torch.tensor(line_points_list).float().cuda(self.gpu))
                    second_nearest_idx = self.search_nearest_point(torch.tensor(second_sample).float().cuda(self.gpu),
                                                                   torch.tensor(line_points_list).float().cuda(
                                                                       self.gpu))
                    first_sample_gt = line_points_list[first_nearest_idx, :]
                    second_sample_gt = line_points_list[second_nearest_idx, :]
                    sample_points = np.concatenate((first_sample, second_sample), axis=1)
                    sample_gt = np.concatenate((first_sample_gt, second_sample_gt), axis=1)
                    sample.append(sample_points)
                    sample_near.append(sample_gt)

                # concatenate
                sample = np.concatenate(sample, axis=0)
                sample_near = np.concatenate(sample_near, axis=0)
                binary_flags = self.compute_flag(sample, edges)
                udf = self.compute_udf(torch.tensor(sample).float().cuda(self.gpu),
                                       torch.tensor(sample_near).float().cuda(self.gpu))
                # for idx in range(sample.shape[0]):
                #     # if idx >= self.sample_num * 0.9:
                #     #     print(idx)
                #         # cv2.circle(img, tuple([int(sample[idx, 0]), int(sample[idx, 1])]), 1, [255, 255, 255], -1)
                #         # cv2.circle(img, tuple([int(sample[idx, 2]), int(sample[idx, 3])]), 1, [0, 255, 0], -1)
                #     if binary_flags[idx]:
                #         # print('yes')
                #         cv2.line(img, tuple([int(sample[idx, 0]), int(sample[idx, 1])]), tuple([int(sample[idx, 2]), int(sample[idx, 3])]), [0, 0, 255], 1)
                #     # else:
                #     #     cv2.line(img, tuple([int(sample[idx, 0]), int(sample[idx, 1])]), tuple([int(sample[idx, 2]), int(sample[idx, 3])]), [0, 255, 0], 1)
                #         # cv2.circle(img, tuple([int(sample_near[idx, 0]), int(sample_near[idx, 1])]), 1,
                #         #            [0, 0, 255], -1)
                #         # print(idx)
                #         # print(udf[idx])
                # cv2.imshow('img', img)
                # cv2.waitKey(0)
            else:
                sample_point = []
                sample_point_idx = np.random.choice(256 * 256, int(self.sample_num), replace=False)
                for idx in range(len(sample_point_idx)):
                    point_idx = sample_point_idx[idx]
                    pt_x, pt_y = int(point_idx / 256), point_idx % 256
                    sample_point.append([pt_x, pt_y])
                    # sample = np.vstack((sample, np.array([pt_x, pt_y], dtype=np.float64)))
                # print(sample_point)
                sample_point = np.array(sample_point, dtype=np.float64)
                # surface_sample_scales = [1, 2, 4, 8, 16, 32]
                # surface_sample_ratios = [0.25, 0.25, 0.25, 0.125, 0.0625, 0.0625]  # sum: 0.9
                sample = []
                sample_near = []
                for sample_ratio, sample_scale in zip(self.surface_sample_ratios, self.surface_sample_scales):
                    sample_points_num = int(self.sample_num * sample_ratio)
                    select_indices = np.random.choice(len(sample_point), sample_points_num)
                    select_list = sample_point[select_indices]
                    random_pairs = np.tile(select_list, (1, 2))
                    assert random_pairs.shape[1] == 4  # shape: N x 4
                    random_pairs = random_pairs + np.random.randn(*random_pairs.shape) * sample_scale
                    random_pairs[random_pairs < 0] = 0
                    random_pairs[random_pairs > 255] = 255
                    first_sample, second_sample = random_pairs[:, 0:2], random_pairs[:, 2:4]
                    first_nearest_idx = self.search_nearest_point(torch.tensor(first_sample).float().cuda(self.gpu),
                                                                  torch.tensor(line_points_list).float().cuda(self.gpu))
                    second_nearest_idx = self.search_nearest_point(torch.tensor(second_sample).float().cuda(self.gpu),
                                                                   torch.tensor(line_points_list).float().cuda(
                                                                       self.gpu))
                    first_sample_gt = line_points_list[first_nearest_idx, :]
                    second_sample_gt = line_points_list[second_nearest_idx, :]
                    sample_points = np.concatenate((first_sample, second_sample), axis=1)
                    sample_gt = np.concatenate((first_sample_gt, second_sample_gt), axis=1)
                    sample.append(sample_points)
                    sample_near.append(sample_gt)
                sample = np.concatenate(sample, axis=0)
                sample_near = np.concatenate(sample_near, axis=0)
                udf = self.compute_udf(torch.tensor(sample).float().cuda(self.gpu),
                                       torch.tensor(sample_near).float().cuda(self.gpu))
                binary_flags = self.compute_flag(sample, edges)
            assert len(sample) == self.sample_num
            assert len(binary_flags) == self.sample_num
            assert len(udf) == self.sample_num
            udf = udf / 255.0
            end = time.time()
            print('time: ', end - start)
            # convert sample to  -1 - 1
            # sample = self.transform_sample_coord(sample)
            label_name = file_name + '.npz'
            label_path = os.path.join(self.label_folder, label_name)
            np.savez(label_path, grid_coords=sample, labels=binary_flags, df=udf)

    def transform_sample_coord(self, sample):
        shape = [256, 256]
        for i, n in enumerate(shape):
            v0, v1 = -1, 1
            r = (v1 - v0) / (2 * n)
            sample[:, i] = v0 + r + (2 * r) * sample[:, i]
            # sample[:, i+2] = v0 + r + (2 * r) * sample[:, i+2]
        return sample

    def preprocess_img(self, is_training):
        self.img_folder = 'preprocess_img'
        if not os.path.exists(self.img_folder):
            os.makedirs(self.img_folder)
        if is_training == 0:
            file_list = self.train_file_list
            print('Process training set')
        elif is_training == 1:
            file_list = self.valid_file_list
            print('Process val set')
        elif is_training == 2:
            file_list = self.test_file_list
            print('Process test set')
        for file_name in file_list:
            print(file_name)
            annot_path = os.path.join(self.annotation_path, file_name + '.npy')
            annot = np.load(annot_path, allow_pickle=True, encoding='latin1').tolist()
            img_path = os.path.join(self.density_path, file_name + '.png')
            img = cv2.imread(img_path)
            copy_img = np.zeros(img.shape)
            gt_data = self.convert_annot(annot)
            edges = self.draw_edges(gt_data)
            line_points_list = []
            for edge in edges:
                point0, point1 = edge[0], edge[1]
                lines_points = self.sample_points_on_line_segment(point0, point1, distance=0.1)
                line_points_list.extend(lines_points)
            line_points_list = np.array(line_points_list)
            for idx in range(len(line_points_list)):
                # print(img[int(line_points_list[idx, 0]), int(line_points_list[idx, 1]), :])
                # cv2.circle(img, tuple([int(line_points_list[idx, 0]), int(line_points_list[idx, 1])]), 1, [0, 0, 255], -1)
                # copy_img[int(line_points_list[idx, 1]), int(line_points_list[idx, 0]), :] = img[int(line_points_list[idx, 1]), int(line_points_list[idx, 0]), :]
                copy_img[int(line_points_list[idx, 1]), int(line_points_list[idx, 0]), :] = [255, 255, 255]

            # cv2.imshow('1', copy_img)
            # cv2.imshow('2', img)
            # cv2.waitKey(0)
            save_path = os.path.join(self.img_folder, file_name + '.png')
            cv2.imwrite(save_path, copy_img)
            # print(img.shape)


if __name__ == '__main__':
    runner = Generate_Label()
    # runner.preprocess_img(0)
    # runner.preprocess_img(1)
    # runner.preprocess_img(2)
    # runner.generate_label()
    # runner.convert_grid(0)
    # runner.convert_grid(1)
    # runner.convert_grid(2)
    # runner.generate_udf_label(0)
    # runner.generate_udf_label(1)
    # runner.generate_udf_label(2)
    print('finish')
