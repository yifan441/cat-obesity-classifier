import torch
import torch.nn as nn
import torch.nn.functional as F


class CatClassifier(nn.Module):
    '''
    Requires input size to be 256x256
    Convolutional network to classify cat images as:
    - skinny
    - normal
    - obese
    '''
    def __init__(self, dropout=0.5):
        super().__init__()
        self.maxpool = nn.MaxPool2d(
            kernel_size=2
        )
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=7
        )
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5
        )
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3
        )
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(107648, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 3)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return F.softmax(x)

