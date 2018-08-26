import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, inchannel, hiddenchannel, outchannel, inlinear=8, stride=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=hiddenchannel, stride=stride, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hiddenchannel, out_channels=outchannel, stride=stride, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2)
        )
        self.fc1 = nn.Linear(outchannel * 8 * 8, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        output = self.fc2(x)
        return output
