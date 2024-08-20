import torch
import torch.nn as nn
import torch.nn.functional as F

class CatClassifier(nn.Module):
    '''
    Convolutional network to classify cat images as:
    - skinny
    - normal
    - obese
    '''
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(
            kernel_size=2
        )
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3
        )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3
        )
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=16,
            kernel_size=3
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(14400, 256)
        self.linear2 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return F.softmax(x)

