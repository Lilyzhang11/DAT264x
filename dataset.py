import os
import csv
import cv2
import torch
from torch.utils.data import Dataset as Ds
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np


class Dataset(Ds):

    def __init__(self, img_dir, label_file, train=False):
        self.img_dir = img_dir
        self.train = train

        self.label = []
        with open(label_file, 'r') as f:
            reader = csv.reader(f)
            for idx, (file_id, accent) in enumerate(reader):
                if idx == 0:
                    continue
                self.label.append([file_id, accent])

    def __getitem__(self, idx):
        file_id, accent = self.label[idx]

        # 图片
        if int(file_id) < 20000:
            img_name = os.path.join(self.img_dir, 'train', file_id+'.png')
        else:
            img_name = os.path.join(self.img_dir, 'test', file_id+'.png')
        # img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_name)
        img = self.preprocessing(img)
        if self.train:
            img = self.augment(img)
        img = TF.to_tensor(img)

        # 类别
        label = torch.zeros(3)
        label[int(accent)] = 1.

        return img, label

    def preprocessing(self, img):
        # cv2.imwrite('src.png', img)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        img = cv2.filter2D(img, -1, kernel=kernel)
        # cv2.imwrite('dst.png', img)
        # while True:
        #     pass
        return img

    def augment(self, img):
        img = Image.fromarray(img)
        if torch.rand(1).item() > 0.5:
            img = TF.hflip(img)
        return img

    def __len__(self):
        return len(self.label)


if __name__ == '__main__':

    dataset = Dataset('data/processed/train/', 'data/processed/train.csv')
    img, label = dataset[0]
    print(img.size(), label.size())
