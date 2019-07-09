import os
import csv
import cv2
import torch
from torch.utils.data import Dataset as Ds
import torchvision.transforms.functional as TF


class Dataset(Ds):

    def __init__(self, img_dir, label_file):
        self.img_dir = img_dir

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
        img_name = os.path.join(self.img_dir, file_id+'.png')
        # img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_name)
        img = TF.to_tensor(img)

        # 类别
        label = torch.zeros(3)
        label[int(accent)] = 1.

        return img, label

    def __len__(self):
        return len(self.label)


if __name__ == '__main__':

    dataset = Dataset('data/processed/train/', 'data/processed/train.csv')
    img, label = dataset[0]
    print(img.size(), label.size())