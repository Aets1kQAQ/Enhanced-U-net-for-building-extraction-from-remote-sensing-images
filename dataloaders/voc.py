# Originally written by Kazuto Nakashima 
# https://github.com/kazuto1011/deeplab-pytorch

from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class VOCDataset(BaseDataSet):
    """
    Pascal Voc dataset
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    """
    def __init__(self, **kwargs):
        self.num_classes = 2
        self.palette = palette.get_voc_palette(self.num_classes)
        super(VOCDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root, 'building')
        self.image_dir = os.path.join(self.root, 'JPEGImages')
        self.label_dir = os.path.join(self.root, 'SegmentationClass')

        file_list = os.path.join(self.root, "ImageSets/Segmentation", self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]

    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.tif')
        label_path = os.path.join(self.label_dir, image_id + '.tif')
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        label = np.where(label == 255, 1, label)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label, image_id

    class VOCDataset(torch.utils.data.Dataset):
        def __init__(self, data_dir, transform=None):
            self.data_dir = data_dir
            self.transform = transform
            # 加载你的图像和标签路径列表
            # self.images = ...
            # self.labels = ...

        def __getitem__(self, index):
            # 读取图像和标签
            image = self._load_image(self.images[index])
            label = self._load_label(self.labels[index])

            # 将标签中值为 255 的部分替换为 1
            label = self._convert_label(label)

            if self.transform is not None:
                image, label = self.transform(image, label)

            return image, label

        def __len__(self):
            return len(self.images)

        def _load_image(self, image_path):
            # 读取图像的函数
            image = ...  # 读取图像代码
            return image

        def _load_label(self, label_path):
            # 读取标签的函数
            label = ...  # 读取标签代码
            return label

        def _convert_label(self, label):
            if isinstance(label, torch.Tensor):
                label = label.numpy()  # 如果是 Tensor，先转换为 numpy 数组
            # 假设有2个类别，灰度值0表示背景，1表示前景
            # 如果你的标签灰度值超过两个类别，可以根据具体情况调整此映射
            label[label > 1] = 1  # 将大于1的标签值都设为前景类（你可以根据实际情况调整）
            return torch.from_numpy(label)  # 转回 Tensor


class VOCAugDataset(BaseDataSet):
    """
    Contrains both SBD and VOC 2012 dataset
    Annotations : https://github.com/DrSleep/tensorflow-deeplab-resnet#evaluation
    Image Sets: https://ucla.app.box.com/s/rd9z2xvwsfpksi7mi08i2xqrj7ab4keb/file/55053033642
    """
    def __init__(self, **kwargs):
        self.num_classes = 2
        self.palette = palette.get_voc_palette(self.num_classes)
        super(VOCAugDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root, 'building')

        file_list = os.path.join(self.root, "ImageSets/Segmentation", self.split + ".txt")
        file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
        self.files, self.labels = list(zip(*file_list))
    
    def _load_data(self, index):
        image_path = os.path.join(self.root, self.files[index][1:])
        label_path = os.path.join(self.root, self.labels[index][1:])
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        label = np.where(label == 255, 1, label)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label, image_id


class VOC(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):
        
        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }
    
        if split in ["train_aug", "trainval_aug", "val_aug", "test_aug"]:
            self.dataset = VOCAugDataset(**kwargs)
        elif split in ["train", "trainval", "val", "test"]:
            self.dataset = VOCDataset(**kwargs)
        else: raise ValueError(f"Invalid split name {split}")
        super(VOC, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

