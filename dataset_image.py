import os
import random
import numpy as np
import PIL.Image
import torch
from torch.utils import data
import cv2
import pdb


class MyData(data.Dataset):
    """
    load images for testing
    root: director/to/images/
            structure:
            - root
                - images (images here)
                - masks (ground truth)
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def __init__(self, root, transform=True):
        super(MyData, self).__init__()
        self.root = root
        self.is_transform = transform

        img_root = os.path.join(self.root, 'DUTS-TR-Image')
        gt_root = os.path.join(self.root, 'test_val3')
        file_names = os.listdir(gt_root)
        self.img_names = []
        self.gt_names = []
        self.names = []
        for i, name in enumerate(file_names):
            if not name.endswith('.png'):
                continue
            self.img_names.append(
                os.path.join(img_root, name[:-4] + '.jpg')
            )
            self.gt_names.append(
                os.path.join(gt_root, name[:-4] + '.png')
            )
            self.names.append(name[:-4])

    def __len__(self):
        return len(self.gt_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        if len(img.shape) < 3:
            img = np.stack((img, img, img), 2)
        if img.shape[2] > 3:
            img = img[:, :, :3]

        gt_file = self.gt_names[index]
        gt = PIL.Image.open(gt_file)
        gt = np.array(gt, dtype=np.int32)
        gt[gt != 0] = 1

        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        gt = cv2.resize(gt, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

        if self.is_transform:
            img, gt = self.transform(img, gt)
            return img, gt
        else:
            return img, gt

    def transform(self, img, gt):
        img = img.astype(np.float64) / 255
        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        gt = gt.astype(np.float32)
        gt = torch.from_numpy(gt)
        return img, gt


class MyTestData(data.Dataset):
    """
    load images for testing
    root: director/to/images/
            structure:
            - root
                - images
                    - images (images here)
                - masks (ground truth)
                - ptag (initial maps)
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def __init__(self, root, name, transform=True):
        super(MyTestData, self).__init__()
        self.root = root
        self.pior = name
        self._transform = transform
        img_root = os.path.join(self.root, self.pior)   #

        file_names = os.listdir(img_root)
        self.img_names = []
        self.names = []
        for i, name in enumerate(file_names):

            if not name.endswith('.jpg'):
                continue
            self.img_names.append(
                os.path.join(img_root, name[:-4] + '.jpg')
            )
            self.names.append(name[:-4])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        ori_img = PIL.Image.open(img_file)
        img_size = ori_img.size
        ori_img = np.array(ori_img, dtype=np.uint8)

        if len(ori_img.shape) < 3:
            ori_img = np.stack((ori_img, ori_img, ori_img), 2)
        if ori_img.shape[2] > 3:
            ori_img = ori_img[:, :, :3]

        img = cv2.resize(ori_img, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        img = np.array(img, dtype=np.uint8)

        if self._transform:
            img, ori_img = self.transform(img, ori_img)
            return img, ori_img, self.names[index], img_size
        else:
            return img, ori_img, self.names[index], img_size

    def transform(self, img, ori_img):
        img = img.astype(np.float64) / 255
        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)
        ori_img = ori_img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        ori_img = ori_img.astype(np.float64)
        ori_img = torch.from_numpy(ori_img).float()

        return img, ori_img


class CRF_Data(data.Dataset):

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def __init__(self, root, transform=True):
        super(CRF_Data, self).__init__()
        self.root = root
        self.is_transform = transform

        img_root = os.path.join(self.root, 'DUTS-TR-Image')
        gt_root = os.path.join(self.root, 'test_val3')
        file_names = os.listdir(gt_root)
        self.img_names = []
        self.gt_names = []
        self.names = []
        for i, name in enumerate(file_names):

            self.img_names.append(
                os.path.join(img_root, name[:-4] + '.jpg')
            )
            self.gt_names.append(
                os.path.join(gt_root, name[:-4] + '.png')
            )
            self.names.append(name[:-4])

    def __len__(self):
        return len(self.gt_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        ori_img = PIL.Image.open(img_file)
        img_size = ori_img.size
        ori_img = np.array(ori_img , dtype=np.uint8)

        if len(ori_img .shape) < 3:
            ori_img = np.stack((ori_img , ori_img , ori_img ), 2)
        if ori_img .shape[2] > 3:
            ori_img = ori_img [:, :, :3]

        gt_file = self.gt_names[index]
        gt = PIL.Image.open(gt_file)
        gt = np.array(gt, dtype=np.int32)

        '''if gt.shape[2] > 3:
            gt = gt[:, :, :1]'''
        #gt[gt != 0] = 1

        #ori_img = cv2.resize(ori_img, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        #gt = cv2.resize(gt, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

        if self.is_transform:
            ori_img, gt = self.transform(ori_img, gt)
            return ori_img, gt, self.names[index], img_size
        else:
            return ori_img, gt,  self.names[index], img_size

    def transform(self, ori_img, gt):
        img = ori_img.astype(np.float64) / 255
        #img -= self.mean
        #img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        gt = gt.astype(np.int32)
        gt = torch.from_numpy(gt)

        return img, gt