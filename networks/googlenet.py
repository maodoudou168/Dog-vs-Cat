import torch.nn as nn
import torch
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, inchannels, outchannels, **kwargs):
        super(BasicConv2d, self).__init__(),
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=outchannels, bias=False, **kwargs),
            nn.BatchNorm2d(num_features=outchannels, eps=1e-5),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Inception(nn.Module):
    def __init__(self, cep_in, cep_1x1_out, cep_3x3_mid, cep_3x3_out, cep_5x5_mid, cep_5x5_out, cep_pool_mid,
                 cep_pool_out):
        super(Inception, self).__init__(),
        self.inception_1x1 = BasicConv2d(inchannels=cep_in, outchannels=cep_1x1_out, kernel_size=1, stride=1)

        self.inception_3x3 = nn.Sequential(
            BasicConv2d(inchannels=cep_in, outchannels=cep_3x3_mid, kernel_size=1, stride=1),
            BasicConv2d(inchannels=cep_3x3_mid, outchannels=cep_3x3_out, kernel_size=3, stride=1, padding=1),
        )

        self.inception_5x5 = nn.Sequential(
            BasicConv2d(inchannels=cep_in, outchannels=cep_5x5_mid, kernel_size=1, stride=1),
            BasicConv2d(inchannels=cep_5x5_mid, outchannels=cep_5x5_out, kernel_size=5, stride=1, padding=2),
        )

        self.inception_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1),
            BasicConv2d(inchannels=cep_pool_mid, outchannels=cep_pool_out, kernel_size=1, stride=1, padding=1),
        )

    def forward(self, x):
        x_1x1 = self.inception_1x1(x)
        x_3x3 = self.inception_3x3(x)
        x_5x5 = self.inception_5x5(x)
        x_pool = self.inception_pool(x)
        outputs = [x_1x1, x_3x3, x_5x5, x_pool]
        return torch.cat(outputs, 1)


# class InceptionClassifier(nn.Module):
#     def __init__(self, cep_in, infeatures, num_class):
#         super(InceptionClassifier, self).__init__(),
#         self.Conv = nn.Sequential(
#             nn.AvgPool2d(kernel_size=5, stride=3),
#             BasicConv2d(inchannels=cep_in, outchannels=128),
#             nn.ReLU(),
#         )
#         self.fc1 = nn.Sequential(
#             nn.Linear(in_features=infeatures, out_features=1024),
#             nn.ReLU(),
#             nn.Dropout()
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(in_features=1024, out_features=num_class),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Softmax2d()
#         )
#
#     def forward(self, x):
#         x = self.Conv(x)
#         x = self.fc(x)




class GoogleNet(nn.Module):
    def __init__(self, num_class):
        super(GoogleNet, self).__init__(),

        self.Conv1 = nn.Sequential(
            BasicConv2d(inchannels=3, outchannels=64, kernel_size=7, stride=2, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.Conv2 = nn.Sequential(
            BasicConv2d(inchannels=64, outchannels=64, kernel_size=1, stride=1, padding=1),
            BasicConv2d(inchannels=64, outchannels=192, kernel_size=3, stride=1, padding=1)
        )

        self.Conv3a = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            Inception(cep_in=192, cep_1x1_out=64, cep_3x3_mid=96, cep_3x3_out=128, cep_5x5_mid=16, cep_5x5_out=32,
                      cep_pool_mid=192, cep_pool_out=32)
        )

        self.Conv3b = Inception(cep_in=256, cep_1x1_out=128, cep_3x3_mid=128, cep_3x3_out=192, cep_5x5_mid=32,
                                cep_5x5_out=96, cep_pool_mid=256, cep_pool_out=64)

        # 添加辅助分类器1

        self.Conv4a = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            Inception(cep_in=480, cep_1x1_out=192, cep_3x3_mid=96, cep_3x3_out=208, cep_5x5_mid=16, cep_5x5_out=48,
                      cep_pool_mid=480, cep_pool_out=64)
        )

        self.Conv4b = Inception(cep_in=512, cep_1x1_out=160, cep_3x3_mid=112, cep_3x3_out=224, cep_5x5_mid=24,
                                cep_5x5_out=64, cep_pool_mid=512, cep_pool_out=64)

        self.Conv4c = Inception(cep_in=512, cep_1x1_out=128, cep_3x3_mid=128, cep_3x3_out=256, cep_5x5_mid=24,
                                cep_5x5_out=64, cep_pool_mid=512, cep_pool_out=64)

        self.Conv4d = Inception(cep_in=512, cep_1x1_out=112, cep_3x3_mid=144, cep_3x3_out=288, cep_5x5_mid=32,
                                cep_5x5_out=64, cep_pool_mid=512, cep_pool_out=64)

        self.Conv4e = Inception(cep_in=528, cep_1x1_out=256, cep_3x3_mid=160, cep_3x3_out=320, cep_5x5_mid=32,
                                cep_5x5_out=128, cep_pool_mid=528, cep_pool_out=128)

        # 添加辅助分类器2

        self.Conv5a = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            Inception(cep_in=832, cep_1x1_out=256, cep_3x3_mid=160, cep_3x3_out=320, cep_5x5_mid=32, cep_5x5_out=128,
                      cep_pool_mid=832, cep_pool_out=128)
        )

        self.Conv5b = Inception(cep_in=832, cep_1x1_out=384, cep_3x3_mid=192, cep_3x3_out=384, cep_5x5_mid=48,
                                cep_5x5_out=128, cep_pool_mid=832, cep_pool_out=128)

        self.Classifier = nn.Linear(in_features=1024, out_features=num_class)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3a(x)
        x = self.Conv3b(x)
        x = self.Conv4a(x)
        x = self.Conv4b(x)
        x = self.Conv4c(x)
        x = self.Conv4d(x)
        x = self.Conv4e(x)
        x = self.Conv5a(x)
        x = self.Conv5b(x)
        x = F.relu(F.avg_pool2d(x, kernel_size=7, stride=1))
        x = F.dropout(x, 0.4)
        x = self.Classifier(out_features=1000)


