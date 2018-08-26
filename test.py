import torch
from torchvision import transforms
from data.dogcat import DogCat
from torch.utils import data
from networks.cnnnet import CNN
from processing_functions import test, catalogue

resize_val = 32
# ##############################建立测试数据集
# 设置transform
test_transform = transforms.Compose([
    transforms.Resize(resize_val),
    transforms.CenterCrop(resize_val),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 将测试集装入testset
testset = DogCat('./data/test', train=False, transform=test_transform)

# 创建迭代器
testloader = data.DataLoader(testset, 30, False, num_workers=2)

model = CNN(inchannel=3, hiddenchannel=64, outchannel=128)

model.load_state_dict(torch.load('./data/model.pth'))


def main():
    result = test(testloader, model)
    catalogue(result)


if __name__ == '__main__':
    main()

