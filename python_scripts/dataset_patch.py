# coding:utf-8

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/7/21 11:20 AM
"""
import copy
import math
import numpy as np
import cv2
from utils.augmentations import letterbox


class LoadImage2Patch:
    # def __init__(self, img_size=(3000, 4096), patch_size=(3000, 4096), overlapping_size=(0, 0)):
    def __init__(self, img_size=(3000, 4096), patch_size=(1500, 2048), overlapping_size=(0, 0)):
        self.img_size = img_size
        self.d_x, self.d_y = int(patch_size[1]), int(patch_size[0])
        self.o_x, self.o_y = int(overlapping_size[1]), int(overlapping_size[0])
        self.cell_width, self.cell_height = self.d_x + self.o_x, self.d_y + self.o_y
        self.top_left_corner_coord = self.compute_top_left_corners()
        self.small_bbox_side_ignore = 1

    def compute_top_left_corners(self):
        len_y, len_x = self.img_size
        top_left_corner_coord = []
        num_cell_y, num_cell_x = math.ceil(len_y / self.d_y), math.ceil(len_x / self.d_x)
        for i in range(num_cell_y):
            top_left_corner_coord_i = []
            for j in range(num_cell_x):

                # extend half of overlap to all direction
                index_y = i * self.d_y - int(self.o_y / 2)
                if i == 0:  # extend overlap down
                    index_y = i * self.d_y
                elif i == num_cell_y - 1:  # extend overlap up
                    index_y = i * self.d_y - self.o_y

                index_x = j * self.d_x - int(self.o_x / 2)
                if j == 0:
                    index_x = j * self.d_x
                elif j == num_cell_x - 1:
                    index_x = j * self.d_x - self.o_x

                top_left_corner_coord_i.append((index_y, index_x))
            top_left_corner_coord.append(top_left_corner_coord_i)
        return top_left_corner_coord

    def img_2_cell(self, img):
        assert img.shape[:2] == self.img_size, 'expect ' + str(self.img_size) + ' got ' + str(img.shape[:2])
        cells = [[None for j in range(len(self.top_left_corner_coord[0]))] for i in
                 range(len(self.top_left_corner_coord))]
        for i, y_x_row in enumerate(self.top_left_corner_coord):
            for j, (y, x) in enumerate(y_x_row):
                cells[i][j] = img[y:y + self.d_y + self.o_y, x:x + self.d_x + self.o_x, :]
        return cells

    def labels_from_img_coord_2_cell_coord(self, labels):
        bboxs_in_cell = [[[] for j in range(len(self.top_left_corner_coord[0]))] for i in
                         range(len(self.top_left_corner_coord))]
        for label in labels:
            cls, bbox = label[0], label[1:]
            # check each cell
            for i, y_x_row in enumerate(self.top_left_corner_coord):
                for j, (y, x) in enumerate(y_x_row):
                    x_min, y_min, x_max, y_max = bbox
                    x_min_cell, y_min_cell, x_max_cell, y_max_cell = max(x, x_min), \
                                                                     max(y, y_min), \
                                                                     min(x + self.d_x + self.o_x - 1, x_max), \
                                                                     min(y + self.d_y + self.o_y - 1, y_max)
                    # if enough part of bbox in this cell
                    if x_min_cell + self.small_bbox_side_ignore < x_max_cell and y_min_cell + self.small_bbox_side_ignore < y_max_cell:
                        bboxs_in_cell[i][j].append(
                            [cls, x_min_cell - x, y_min_cell - y, x_max_cell - x, y_max_cell - y])
        return bboxs_in_cell

    def img_and_label_from_img_2_cell(self, img, label):
        """"""
        img_cell = self.img_2_cell(img)
        bbox_in_cell = self.labels_from_img_coord_2_cell_coord(label)
        return img_cell, bbox_in_cell

    def example_cells(self):
        img = np.arange(0, self.img_size[0] * self.img_size[1])
        img = img.reshape((self.img_size[0], self.img_size[1], -1))
        img = np.concatenate([img, img, img], axis=-1)
        img_cells = self.img_2_cell(img)
        return img_cells


class LoadImg2Patch:
    def __init__(self):
        self.img_size = 640
        self.stride = 16

        self.patcher = LoadImage2Patch()

        self.len = 0
        self.patches, self.origins = [], []

        self.img = None
        self.patches_org = None

    def load(self, img):
        self.img = img
        patch_2d = self.patcher.img_2_cell(img)
        origin_2d = self.patcher.compute_top_left_corners()
        self.patches_org, self.patches, self.origins = [], [], []
        for patch_row, origin_row in zip(patch_2d, origin_2d):
            for patch_org, origin in zip(patch_row, origin_row):
                # Padded resize
                patch = letterbox(patch_org, self.img_size, stride=self.stride)[0]
                # patch = np.ascontiguousarray(patch)
                self.patches.append(patch)
                self.patches_org.append(patch_org)
                self.origins.append(origin)

        self.len = len(self.patches)
        self.patches_org = self.patches_org
        self.patches = np.asarray(self.patches)
        self.origins = np.asarray(self.origins)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.len:
            raise StopIteration
        img = self.patches[self.count, :]
        origin = self.origins[self.count, :]
        assert img is not None, f'Image Empty'

        self.count += 1

        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        return img, origin

    def __len__(self):
        return self.len


class LoadImg2PatchBatch(LoadImg2Patch):
    def __init__(self, batch_size=4):
        super(LoadImg2PatchBatch, self).__init__()
        self.batch_size = batch_size

    def __next__(self):
        if self.count >= self.len:
            raise StopIteration
        imgs_org = self.patches_org[self.count:min(self.count+self.batch_size, self.len)]
        imgs = self.patches[self.count:min(self.count+self.batch_size, self.len), :]
        origins = self.origins[self.count:min(self.count+self.batch_size, self.len)]
        assert imgs is not None, f'Image Empty'

        self.count += self.batch_size

        imgs = imgs.transpose((0, -1, 1, 2))[:, ::-1]  # HWC to CHW, BGR to RGB
        imgs = np.ascontiguousarray(imgs)

        return imgs, imgs_org, copy.deepcopy(self.img), origins


def main():
    img_path = '/home/cheng/proj/data/data_matrix/no/img/13/Image_20211126161810608.bmp'
    img = cv2.imread(img_path)
    dataset = LoadImg2PatchBatch()
    dataset.load(img)
    for batch in dataset:
        imgs, origins = batch
        for img, origin in zip(imgs, origins):
            img = img[::-1].transpose((1, 2, 0))

            cv2.imshow('img', img)
            cv2.waitKey(1000)
            print(origin)


if __name__ == '__main__':
    main()

