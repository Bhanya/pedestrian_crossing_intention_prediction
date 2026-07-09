"""

@ Description:
@ Project:APCIL
@ Author:qufang
@ Create:2024/6/4 21:42

"""
# This code prepares the PIE/JAAD pedestrian images so EfficientPIE can train using one cropped image per pedestrian sequence.
# It takes a pedestrian video sequence, selects one frame, crops the image around the pedestrian, prepares it as a tensor, and
# returns it with the crossing/not-crossing label.

import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

class MyDataSet(Dataset):

    def __init__(
            self,
            images_seq: list,
            data_opts: dict,
            transform=None,
            step=None,
            return_bbox_features=False,
            return_bbox_trajectory=False):
        self.images_seq = images_seq
        self.data_opts = data_opts
        self.transform = transform
        self.step = step
        self.return_bbox_features = return_bbox_features
        self.return_bbox_trajectory = return_bbox_trajectory

    def __len__(self):
        """
        it is the number of sequence, each sequence consists of images,
        the number of images of one sequence is ['max_size_observe']
        :return: the length of train/val/test set
        """
        length = len(self.images_seq['images'])
        return length

    # gets image paths, bounding boxes, and labels for one 15-frame sequence.
    def __getitem__(self, index):
        each_seq_imgs = self.images_seq['images'][index]
        each_seq_bboxes = self.images_seq['bboxes'][index]
        each_seq_labels = self.images_seq['output'][index]

        # label = torch.as_tensor(each_seq_labels[0])
        # sequence = []
        # flip = random.random() > 0.5  # 随机水平翻转
        # brightness_factor = random.uniform(0.5, 1.5)  # 随机亮度
        # contrast_factor = random.uniform(0.5, 1.5)  # 随机对比度
        # saturation_factor = random.uniform(0.5, 1.5)  # 随机饱和度
        # hue_factor = random.uniform(-0.1, 0.1)  # 随机色调
        #
        # for i in range(0, self.data_opts['max_size_observe'], 2):
        #     single_img = each_seq_imgs[i]
        #     single_bbox = each_seq_bboxes[i]
        #
        #     single_img = Image.open(single_img)
        #     if single_img.mode != 'RGB':
        #         raise ValueError("image: {} isn't RGB mode.".format(single_img))
        #     # crop the img,pay attention to the context of bbox
        #     # compute the new location
        #     x1, y1, x2, y2 = single_bbox
        #     center_x = (x1 + x2) / 2
        #     center_y = (y1 + y2) / 2
        #     half_size = 150  # 150
        #     new_x1 = int(center_x - half_size)
        #     new_y1 = int(center_y - half_size)
        #     new_x2 = int(center_x + half_size)
        #     new_y2 = int(center_y + half_size)
        #     # boundary condition
        #     img_width, img_height = single_img.size
        #     new_x1 = max(0, new_x1)
        #     new_y1 = max(0, new_y1)
        #     new_x2 = min(img_width, new_x2)
        #     new_y2 = min(img_height, new_y2)
        #     # crop
        #     crop_box = [new_x1, new_y1, new_x2, new_y2]
        #     single_img = single_img.crop(crop_box)
        #
        #     if flip:
        #         single_img = F.hflip(single_img)  # 随机水平翻转
        #     single_img = F.adjust_brightness(single_img, brightness_factor)
        #     single_img = F.adjust_contrast(single_img, contrast_factor)
        #     single_img = F.adjust_saturation(single_img, saturation_factor)
        #     single_img = F.adjust_hue(single_img, hue_factor)
        #
        #     single_img = F.resize(single_img, [300, 300])
        #     # 转为张量并归一化
        #     single_img = F.to_tensor(single_img)
        #     single_img = F.normalize(single_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        #
        #     # if self.transform is not None:
        #     #     single_img = self.transform(single_img)
        #
        #     sequence.append(single_img)
        # sequence = torch.cat(sequence, dim=0)
        # return sequence, label

        reverse_step = 1 # select the last frame from the 15-frame sequence
        last_img = each_seq_imgs[self.data_opts['max_size_observe'] - reverse_step]
        last_bbox = each_seq_bboxes[self.data_opts['max_size_observe'] - reverse_step]
        last_label = each_seq_labels[self.data_opts['max_size_observe'] - reverse_step]

        img = Image.open(last_img)
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(last_img))
        # crop the img,pay attention to the context of bbox
        # compute the new location
        x1, y1, x2, y2 = last_bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        half_size = 150  # 150
        new_x1 = int(center_x - half_size)
        new_y1 = int(center_y - half_size)
        new_x2 = int(center_x + half_size)
        new_y2 = int(center_y + half_size)
        # boundary condition
        img_width, img_height = img.size
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(img_width, new_x2)
        new_y2 = min(img_height, new_y2)
        # crop
        crop_box = [new_x1, new_y1, new_x2, new_y2]
        img = img.crop(crop_box)

        if self.transform is not None:
            img = self.transform(img)
        label = torch.as_tensor(last_label)

        # bbox - Bounding box
        # Version 2: creates the 15-frame bbox trajectory.
        if self.return_bbox_trajectory:
            bbox_trajectory = []
            for bbox in each_seq_bboxes[:self.data_opts['max_size_observe']]:
                bx1, by1, bx2, by2 = bbox
                bbox_trajectory.append([
                    ((bx1 + bx2) / 2) / img_width,
                    ((by1 + by2) / 2) / img_height,
                    (bx2 - bx1) / img_width,
                    (by2 - by1) / img_height,
                ])
            bbox_trajectory = torch.as_tensor(bbox_trajectory, dtype=torch.float32)
            return img, bbox_trajectory, label

        # Version 1: creates only the last-frame bbox
        if self.return_bbox_features:
            bbox_features = torch.as_tensor([
                center_x / img_width,
                center_y / img_height,
                (x2 - x1) / img_width,
                (y2 - y1) / img_height,
            ], dtype=torch.float32)
            return img, bbox_features, label

        return img, label

    # Batches sample together for training
    @staticmethod
    def collate_fn(batch):
        # batch is like [(img1,label1),(img2,label2),(img3,label3)]
        # *batch:unpacking the batch list
        # zip(*batch) will combine first and second elements separately
        # zip(*batch) will return:images=(image1,image2,image3),labels=......
        # tuple(zip(*batch)) will make the images and labels become tuple
        if len(batch[0]) == 3:
            imgs, bbox_context, labels = tuple(zip(*batch))

            imgs = torch.stack(imgs, dim=0)
            bbox_context = torch.stack(bbox_context, dim=0)
            labels = torch.stack(labels)
            labels = torch.squeeze(labels, dim=1)

            return imgs, bbox_context, labels

        imgs, labels = tuple(zip(*batch))

        imgs = torch.stack(imgs, dim=0)
        labels = torch.stack(labels)
        labels = torch.squeeze(labels, dim=1)

        return imgs, labels
