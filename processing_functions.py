# coding: utf-8
import torch
from torch.autograd import Variable
import os
import shutil


def train(trainloader, epoch, model, optimizer, criterion):
    model.train()
    for batch_idx, (img, label) in enumerate(trainloader):
        # CUDA
        # if torch.cuda.is_available():
        #     img = Variable(img).cuda()
        #     label = Variable(label).cuda()
        # else:
        img = Variable(img)
        label = Variable(label)
        # 向前传递
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        # 反向传播
        loss.backward()
        optimizer.step()
        print('Epoch:{}[{}/{}] loss:{}'.format(epoch, batch_idx, len(trainloader), loss.mean()))


def eva(evaloader, model, criterion, evaset):
    acc = 0
    eva_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(evaloader):
            if torch.cuda.is_available():
                img = Variable(img).cuda()
                label = Variable(label).cuda()
            else:
                img = Variable(img)
                label = Variable(label)
            output = model(img)
            _, predict = torch.max(output, 1)
            acc += predict.eq(label.view_as(predict)).sum().item()
            eva_loss += criterion(output, label)
        eva_loss /= len(evaset)
        print('\nEvaluation Set: Average Loss:{}, Acurracy: {}/{}\n'.format(eva_loss, acc, len(evaset)))


def test(testloader, model):
    result = []
    model.eval()
    with torch.no_grad():
        for img, label in testloader:
            if torch.cuda.is_available():
                img = Variable(img).cuda()
            else:
                img = Variable(img)
            output = model(img)
            _, predict = torch.max(output, 1)  # torch.max返回的是每张图片的两个类的predict值，而调用max
            # 后返回的是最大值和最大值的位置，我们需要的是位置（即类别），因此predict取第二位
            predict = predict.numpy()
            for num in predict:
                result.append(num)

    print(result, '\n')
    #    result = list(result)
    return result


def catalogue(result):
    i = 0
    path_cat = './result/cat'
    path_dog = './result/dog'
    if not os.path.exists(path_cat):
        os.makedirs(path_cat)
    if not os.path.exists(path_dog):
        os.makedirs(path_dog)

    with open('./result/result.txt', 'a') as F:
        for item in result:
            if item == 0:
                F.write(str(i) + '-->cat\n')
            else:
                F.write(str(i) + '-->dog\n')
            i += 1

    for item in os.listdir('./data/test'):
        index = int(item.split('.')[-2])
        cata = result[index-1]
#        print(index, cata)
        if cata == 0:  # 说明是猫
            shutil.copyfile('./data/test\\' + item, './result/cat\\' + item)
        else:  # 说明是狗
            shutil.copyfile('./data/test\\' + item, './result/dog\\' + item)
