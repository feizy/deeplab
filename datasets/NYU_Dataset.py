# -*- coding: utf-8 -*-
# @Time    : 2018/9/21 17:21
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : Voc_Dataset.py
# @Software: PyCharm
import cv2
import argparse
import PIL
import random
#import scipy.io
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2
import os
import torch
import torch.utils.data as data
import torchvision.transforms as ttransforms

import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
from PIL import Image



class Nyud2_Dataset(data.Dataset):
    def __init__(self,
                 root_path='/home/feizy/datasets/nyuv2/',
                 dataset='NYUD2',
                 base_size=(640,480),
                 crop_size=(640,480),
                 is_training=True):
        """

        :param root_path:
        :param dataset:
        :param base_size:
        :param is_trainging:
        :param transforms:
        """

        self.dataset = dataset
        self.is_training = is_training
        self.base_size = base_size
        self.crop_size = crop_size

        f = h5py.File('/home/feizy/datasets/nyuv2/nyu_depth_v2_labeled.mat')
        self.images = f["images"]
        self.images = np.array(self.images, dtype='float32')
        depths = f["depths"]
        depths = np.array(depths, dtype='float32')
        labels = f["labels"]
        labels = np.array(labels, dtype='float32')
        num = len(labels)
        offset = int(num * 0.8)
        l = [i for i in range(num)]
        random.shuffle(l)
        train_list = l[:offset]
        train_list = sorted(train_list)
        val_list = l[offset:]
        val_list = sorted(val_list)
        self.train_images = self.images[train_list]
        self.train_depths = depths[train_list]
        self.train_labels = labels[train_list]
        self.val_images = self.images[val_list]
        self.val_depths = depths[val_list]
        self.val_labels = labels[val_list]







    def __getitem__(self, index):

        # if self.is_training:
        #     image, label, depth = self._train_sync_transform(self.train_images[index], self.train_labels[index], self.train_depths[index])
        #
        # else:
        #     image, label, depth = self._val_sync_transform(self.val_images[index], self.val_labels[index], self.val_depths[index])
        #
        #
        # return image, label, depth
        if self.is_training:
            return self.train_images[index], self.train_labels[index], self.train_depths[index]
        else:
            return self.val_images[index], self.val_labels[index], self.val_depths[index]
    def _border(self, border):
        if isinstance(border, tuple):
            if len(border) == 2:
                left, top = right, bottom = border
            elif len(border) == 4:
                left, top, right, bottom = border
        else:
            left = top = right = bottom = border
        return left, top, right, bottom

    def _color(self, color, mode):
        if isStringType(color):
            from . import ImageColor

            color = ImageColor.getcolor(color, mode)
        return color
    def _expand(self, image, w, h, border=0, fill=0):
        left, top, right, bottom = self._border(border)
        width = left + w + right
        height = top + h + bottom
        out = Image.new(image.mode, (width, height), self._color(fill, 'RGB'))
        out.paste(image, (left, top))
        return out
    def _train_sync_transform(self, img, mask, depth):
        '''

        :param image:  PIL input image
        :param gt_image: PIL input gt_image
        :return:
        '''
        # random mirror
        # if random.random() < 0.5:
        #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #     mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        #     depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size[0] * 0.5), int(self.base_size[0] * 2.0))
        w = int(640)
        h = int(480)

        ow = short_size
        oh = int(1.0 * h * ow / w)

        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (ow, oh), interpolation=cv2.INTER_NEAREST)
        # pad crop
        # if short_size < crop_size[0]:
        #     padh = crop_size[1] - oh if oh < crop_size[1] else 0
        #     padw = crop_size[0] - ow if ow < crop_size[0] else 0
        #     img = self._expand(img, 640, 480, border=(0, 0, padw, padh), fill=0)
        #     mask = self._expand(mask, 640, 480,border=(0, 0, padw, padh), fill=0)
        #     depth = self._expand(depth, 640, 480,border=(0, 0, padw, padh), fill=0)
        # # random crop crop_size
        # # w = 640
        # # h = 480
        # x1 = random.randint(0, w - crop_size[0])
        # y1 = random.randint(0, h - crop_size[1])
        # img = img.crop((x1, y1, x1 + crop_size[0], y1 + crop_size[1]))
        # mask = mask.crop((x1, y1, x1 + crop_size[0], y1 + crop_size[1]))
        # depth = depth.crop((x1, y1, x1 + crop_size[0], y1 + crop_size[1]))
        # # gaussian blur as in PSP
        # if random.random() < 0.5:
        #     img = img.filter(ImageFilter.GaussianBlur(
        #         radius=random.random()))
        # final transform
        # img, mask, depth = self._img_transform(img), self._mask_transform(mask), self._mask_transform(depth)
        return img, mask, depth

    def _val_sync_transform(self, img, mask, depth):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        # if w > h:
        #     oh = short_size
        #     ow = int(1.0 * w * oh / h)
        # else:
        #     ow = short_size
        #     oh = int(1.0 * h * ow / w)

        ow = short_size[0]
        oh = int(1.0 * h * ow / w)
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (ow, oh), interpolation=cv2.INTER_NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize[0]) / 2.))
        y1 = int(round((h - outsize[1]) / 2.))
        img = img.crop((x1, y1, x1 + outsize[0], y1 + outsize[1]))
        mask = mask.crop((x1, y1, x1 + outsize[0], y1 + outsize[1]))
        depth = depth.crop((x1, y1, x1 + outsize[0], y1 + outsize[1]))
        # final transform
        # img, mask, depth = self._img_transform(img), self._mask_transform(mask), self._mask_transform(depth)
        return img, mask, depth

    def _img_transform(self, image):
        image_transforms = ttransforms.Compose([
            ttransforms.ToTensor(),
            ttransforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        image = image_transforms(image)
        return image

    def _mask_transform(self, gt_image):
        target = np.array(gt_image).astype('int32')
        target = torch.from_numpy(target)
        return target




    def __len__(self):
       if self.is_training:
           return len(self.train_images)
       else:
           return len(self.val_images)
class Nyud2_DataLoader():
    def __init__(self, args):

        self.args = args

        train_set = Nyud2_Dataset(dataset=self.args.dataset,
                                base_size=self.args.base_size,
                                crop_size=self.args.crop_size,
                                is_training=True)
        val_set = Nyud2_Dataset(dataset=self.args.dataset,
                              base_size=self.args.base_size,
                              crop_size=self.args.crop_size,
                              is_training=False)

        self.train_loader = data.DataLoader(train_set,
                                            batch_size=self.args.batch_size,
                                            shuffle=True,
                                            num_workers=self.args.data_loader_workers,
                                            pin_memory=self.args.pin_memory,
                                            drop_last=True)
        self.valid_loader = data.DataLoader(val_set,
                                            batch_size=self.args.batch_size,
                                            shuffle=False,
                                            num_workers=self.args.data_loader_workers,
                                            pin_memory=self.args.pin_memory,
                                            drop_last=True)

        self.train_iterations = (len(train_set) + self.args.batch_size) // self.args.batch_size
        self.valid_iterations = (len(val_set) + self.args.batch_size) // self.args.batch_size



if __name__ == '__main__':
    # arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument('--dataset', default='NYUD2', type=str,
    #                         choices=['voc2012', 'voc2012_aug', 'cityscapes'],
    #                         help='dataset choice')
    # arg_parser.add_argument('--base_size', default=513, type=int,
    #                         help='crop size of image')
    # arg_parser.add_argument('--crop_size', default=513, type=int,
    #                         help='base size of image')
    # arg_parser.add_argument('--num_classes', default=21, type=int,
    #                         help='num class of mask')
    # arg_parser.add_argument('--data_loader_workers', default=2, type=int,
    #                         help='num_workers of Dataloader')
    # arg_parser.add_argument('--pin_memory', default=2, type=int,
    #                         help='pin_memory of Dataloader')
    # arg_parser.add_argument('--split', type=str, default='train',
    #                         help="choose from train/val/test/trainval")
    # args = arg_parser.parse_args()
    # data=scipy.io.loadmat('/home/feizy/datasets/nyu_depth_v2_labeled.mat')
    # print(data['GTcls']["Segmentation"][0,0])
    dataset = Nyud2_Dataset()
    dataloader = data.DataLoader(dataset, batch_size=5)
    for epoch in range(2):
        for data in dataloader:
            x, y, d = data
            print(d)

    # dataloader = Nyud2_DataLoader(args).train_loader
    # print(np.array([[(1,2,3)]]).shape)
    # print(np.array([[np.array(1), np.array(2), np.array(3)]]).shape)

