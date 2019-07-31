import os
import csv
from PIL import Image

import numpy as np
import cv2

import torch
from torch.utils.data import Dataset as Ds
import torchvision.transforms.functional as TF


class Dataset(Ds):

    def __init__(self, img_dir, label_file, train=False, opt={}):
        # img_dir   : 存放图片的路径
        # label_file: 格式同submission_format.csv的.csv文件
        # train     : train只用来决定是否做augment，所以默认为False
        #             因为即使是False，也可以用来做训练，只不过没有
        #             数据增强
        # opt       : 数据增强的超参数
        self.img_dir = img_dir

        self.label = []
        with open(label_file, 'r') as f:
            reader = csv.reader(f)
            for idx, (file_id, accent) in enumerate(reader):
                if idx == 0:
                    continue
                self.label.append([file_id, accent])
        # 此时self.label的格式是[['10000', '0'], ['10001', '1'], ...]

        self.train = train

        default_opt = self.create_default_opt()
        for key in default_opt:
            if key not in opt:
                opt[key] = default_opt[key]
        self.opt = opt

        # 因为去中心化和数据增强无法很好的兼容，所以弃用
        # self.mean = torch.load(img_dir + 'mean.pth')

    def __getitem__(self, idx):
        idx = idx % len(self.label)

        file_id, accent = self.label[idx]

        img_name = os.path.join(self.img_dir, file_id+'.png')

        # 因为resnet必须要rgb图像，所以不用灰度模式打开
        # img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_name)

        if self.train:
            img = self.augment(img, int(accent), self.opt)
        img = TF.to_tensor(img)

        # 生成one-hot向量
        label = torch.zeros(3)
        label[int(accent)] = 1.

        # 此时img为[3*h*w]，label为[3]
        return img, label

    def __len__(self):
        return len(self.label) * self.opt['num_repeat']

    def rand(self):
        # 生成0~1之间的浮点数
        return torch.rand(1).item()

    def randint(self, low, high):
        # 生成属于闭区间[low, high]的整数
        return torch.randint(low=low, high=high+1, size=(1,)).item()

    def create_default_opt(self):
        opt = {}

        # 重复次数，让数据增强可以覆盖得更广
        opt['num_repeat'] = 1

        # 水平翻转的概率
        opt['phflip'] = 0.5

        # 随机裁剪的概率
        opt['crop_probability'] = 0.
        # 裁剪后的宽度
        opt['crop_width'] = 40

        # 进行水平平移的概率
        opt['phtranslation'] = 0.5
        # 向左平移的概率（向右平移的概率同过“1-”计算得到）
        opt['pltranslation'] = 0.5
        # 向左平移的最大幅度, number, left, translation
        opt['nltranslation'] = 86
        # 向右平移的最大幅度, number, right, translation
        opt['nrtranslation'] = 86

        # 开启循环平移的概率
        opt['pcircle'] = 0.5
        # 向左循环平移的概率
        opt['plcircle'] = 0.5
        # 向左循环平移的一组的宽度
        opt['glcircle'] = 8
        # 向左循环平移的最大次数
        opt['nlcircle'] = 10
        # 向右循环平移的一组的宽度
        opt['grcircle'] = 8
        # 向右循环平移的最大次数
        opt['nrcircle'] = 10

        # 开启重叠平移的概率
        opt['pot'] = 0.5
        # 向左重叠平移的概率
        opt['polt'] = 0.5
        # 向左重叠的最大列数
        opt['wolt'] = 20
        # 向左重叠的位置的low和high
        opt['lowolt'] = 0
        opt['higholt'] = 86
        # 向右重叠的最大列数
        opt['wort'] = 20
        # 向右重叠的位置的low和high
        opt['lowort'] = 87
        opt['highort'] = 172

        # 开启随机将某些列置0这项操作的概率
        opt['pvinvalid'] = 0.5
        # 随机置0的列的最大个数
        opt['nvinvalid'] = 17
        # 每次置0的列的最大宽度
        opt['wvinvalid'] = 1

        # 开启随机将某些行置0的这项操作的概率
        opt['phinvalid'] = 0.5
        # 随机置0的最大次数
        opt['nhinvalid'] = 2
        # 随机置0的最大宽度
        opt['whinvalid'] = 10

        return opt

    def augment(self, img, label, opt):
        # 水平翻转
        img = self.hfilp(img, opt)

        # 随机裁剪
        img = self.random_crop(img, opt)
        
        # 水平平移
        img = self.htranslation(img, opt)

        # 循环平移
        img = self.circle_translation(img, opt)

        # 重叠平移
        img = self.over_translation(img, opt)

        # 随机将某些列置0
        img = self.invalid_col(img, opt)

        # 随机将某些行置0
        img = self.invalid_row(img, opt)

        return img

    def hfilp(self, img, opt):
        img = Image.fromarray(img)
        if self.rand() < opt['phflip']:
            img = TF.hflip(img)
        img = np.array(img)
        return img

    def random_crop(self, img, opt):
        if self.rand() < opt['crop_probability']:
            w = opt['crop_width']
            i = self.randint(0, img.shape[1]-w)
            img = img[:, i:i+w, :]
        return img

    def htranslation(self, img, opt):
        h, w = img.shape[0:2]
        if self.rand() < opt['phtranslation']:
            if self.rand() < opt['pltranslation']:
                n = self.randint(1, opt['nltranslation'])
                img[:, 0:w-n, :] = img[:, n:w, :]
                img[:, w-n:w, :] = 0
            else:
                n = self.randint(1, opt['nrtranslation'])
                img[:, n:w, :] = img[:, 0:w-n, :]
                img[:, 0:n, :] = 0
        return img

    def circle_translation(self, img, opt):
        h, w = img.shape[0:2]
        if self.rand() < opt['pcircle']:
            if self.rand() < opt['plcircle']:
                n = opt['glcircle'] * self.randint(1, opt['nlcircle'])
                tmp = img[:, 0:n, :].copy()
                img[:, 0:w-n, :] = img[:, n:w, :]
                img[:, w-n:w, :] = tmp
            else:
                n = opt['grcircle'] * self.randint(1, opt['nrcircle'])
                tmp = img[:, w-n:w, :].copy()
                img[:, n:w, :] = img[:, 0:w-n, :]
                img[:, 0:n, :] = tmp
        return img

    def over_translation(self, img, opt):
        h, w = img.shape[0:2]
        if self.rand() < opt['pot']:
            if self.rand() < opt['polt']:
                l = self.randint(1, opt['wolt'])
                idx = self.randint(opt['lowolt'], opt['higholt'])
                img[:, idx:w-l, :] = img[:, idx+l:w, :]
                img[:, w-l:w, :] = 0
            else:
                l = self.randint(1, opt['wort'])
                idx = self.randint(opt['lowort'], opt['highort'])
                img[:, l:idx, :] = img[:, 0:idx-l, :]
                img[:, 0:l, :] = 0
        return img

    def invalid_col(self, img, opt):
        h, w = img.shape[0:2]
        if self.rand() < opt['pvinvalid']:
            for _ in range(self.randint(0, opt['nvinvalid'])):
                l = self.randint(1, opt['wvinvalid'])
                idx = self.randint(0, w-l)
                img[:, idx:idx+l, :] = 0
        return img

    def invalid_row(self, img, opt):
        h, w = img.shape[0:2]
        if self.rand() < opt['phinvalid']:
            for _ in range(self.randint(0, opt['nhinvalid'])):
                l = self.randint(1, opt['whinvalid'])
                idx = self.randint(0, h-l)
                img[idx:idx+l, :, :] = 0
        return img


if __name__ == '__main__':

    dataset = Dataset(
        'data/processed/train/',
        'data/processed/train_labels.csv',
        train=True
    )

    print('dataset size:', len(dataset))

    os.makedirs('dataset_visual', exist_ok=True)
    for i in range(1000):
        img, label = dataset[0]
        img = TF.to_pil_image(img)
        img.save('dataset_visual/%d.png'%i)

    input('press enter to delete dataset_visual')
    os.system('rm -rf dataset_visual')
