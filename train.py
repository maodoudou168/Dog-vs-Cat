# coding: utf-8
import torch
from networks.cnnnet import CNN
from torchvision import transforms
from data.dogcat import DogCat
from torch.utils import data
from torch import optim
from torch import nn
from processing_functions import train, eva, test

# #####################建立训练数据集dogcat.py
'''定义transform'''
resize_val = 32
train_transform = transforms.Compose([
    transforms.Resize(resize_val),
    # add(err:RuntimeError: inconsistent tensor)
    transforms.CenterCrop(resize_val),
    # transforms.RandomCrop(28),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

'''将猫狗的数据集建立为dataset'''
trainset = DogCat('./data/train', transform=train_transform)

trainloader = data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

# ##########################创建网络的实例
model = CNN(inchannel=3, hiddenchannel=64, outchannel=128)
# CUDA
if torch.cuda.is_available():
    model.cuda()
# print(model)

# #################################建立evaluation数据集
eva_transform = transforms.Compose([
    transforms.Resize(resize_val),
    transforms.RandomCrop(resize_val),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

evaset = DogCat('./data/eva', transform=eva_transform)

evaloader = data.DataLoader(evaset, 10, shuffle=False, num_workers=2)


# ##########################定义优化和损失函数
optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# ##########################开始训练
'''定义train，val，test函数'''


def main():
    for epoch in range(1, 20):
        train(trainloader, epoch, model, optimizer, criterion)
        eva(evaloader, model, criterion, evaset)

    torch.save(model.state_dict(), './data/model.pth')


if __name__ == "__main__":
    main()
