"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import torch as t
from torch.utils.data import Dataset
from data_augmentation import *
import pickle
import copy
import random
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from PIL import Image


def read_image(path, dtype=np.float32, color=True):

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))

def random_flip(img, y_random=False, x_random=False,
                return_param=False, copy=False):

    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img


def inverse_normalize(img):

    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalze(img):

    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()


def preprocess(img, size):

    C, H, W = img.shape
    img = img / 255.
    img = sktsf.resize(img, (C, size, size), mode='reflect')

    normalize = pytorch_normalze

    return normalize(img), (W, H)


class Transform(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, in_data):
        img, label = in_data
        _, H, W = img.shape

        img, size = preprocess(img, self.size)

        img, params = random_flip(
            img, x_random=True, return_param=True)

        return img, label


class COCODataset(Dataset):
    def __init__(self, root_path="data/COCO", year="2014", mode="train", image_size=448, is_training=False):
        if mode in ["train", "val"] and year in ["2014", "2015", "2017"]:
            self.image_path = os.path.join(root_path, "images")
            anno_path = os.path.join(root_path, "anno_pickle", "COCO_{}{}.pkl".format(mode, year))
            id_list_path = pickle.load(open(anno_path, "rb"))
            self.id_list_path = list(id_list_path.values())
        self.classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                        "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                        "teddy bear", "hair drier", "toothbrush"]
        self.class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
                          31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                          55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                          82, 83, 84, 85, 86, 87, 88, 89, 90]

        self.image_size = image_size
        self.num_classes = len(self.classes)
        self.num_images = len(self.id_list_path)
        self.is_training = is_training
        self.tsf = Transform(self.image_size)

    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        image_path = os.path.join(self.image_path, self.id_list_path[item]["file_name"])

        image = read_image(image_path, color=True)
        objects = copy.deepcopy(self.id_list_path[item]["objects"])
        label = list()
        for idx in range(len(objects)):
            objects[idx][4] = self.class_ids.index(objects[idx][4])
            label.append(objects[idx][4])
        if label == []:
            label = [0]

        label = np.stack(label).astype(np.float32)
        image, labels = self.tsf((image, label))

        return image.copy(), labels.copy()
