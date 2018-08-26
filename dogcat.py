# coding: utf-8
import torch.utils.data as data
import os
from PIL import Image

'''定义自己的dataset'''


class DogCat(data.Dataset):
    def __init__(self, root, train=True, transform=None):  # 首先初始化
        self.root = root
        self.train = train
        self.transform = transform
        self.imgs = []
        for img in os.listdir(root):
            self.imgs.append(os.path.join(root, img))

        # imgs = ['./data/train\\cat.0.jpg','./data/train\\cat.1.jpg',......]

    def __getitem__(self, index):  # 装载数据，返回img, label，迭代器中会使用
        img_path = self.imgs[index]  # 将迭代器中每一个图片的路径赋给变量img_path
        if self.train:
            if 'cat' in img_path.split('/')[-1]:  # 判断并将狗分为1类，猫分为0类
                label = 0
            else:
                label = 1
        else:
            label = int(img_path.split("\\")[-1].split('.')[-2])  # 必须转为int形式，否则读取顺序就成了1/10/2/3...
        img = Image.open(img_path)  # 打开图片并保存入img中
        img = self.transform(img)  # 将图片transform
        return img, label

    def __len__(self):
        return len(self.imgs)




