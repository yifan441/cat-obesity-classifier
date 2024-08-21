import torch
import torch.nn as nn
import torch.nn.functional as F


class _conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class downsample(nn.Module):
    """
    Residual connection to allow gradient flow
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.downsample(x)


class _conv_block(nn.Module):
    """
    Convolution blocks with residual connection
    - num_convs convolution blocks
    - one identity transformation (res) added prior to pooling
    - maxpool at the end
    """

    def __init__(self, num_convs, in_channels, out_channels, kernel_size):
        super().__init__()
        self.num_convs = num_convs
        self.conv_0 = _conv(in_channels, out_channels, kernel_size)
        for i in range(1, num_convs):
            setattr(self, f"conv_{i}", _conv(out_channels, out_channels, kernel_size))
        self.res = downsample(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        res = self.res(x)
        for i in range(self.num_convs):
            x = getattr(self, f"conv_{i}")(x)
        x = x + res
        return self.pool(x)


class _fc_block(nn.Module):
    """
    Basic fully-connected block
    """

    def __init__(self, num_layers, dim_in, dim_mid, dim_out):
        super().__init__()
        self.num_layers = num_layers
        self.layer_0 = nn.Linear(dim_in, dim_mid)
        for i in range(1, num_layers - 1):
            setattr(self, f"layer_{i}", nn.Linear(dim_mid, dim_mid))
        self.out = nn.Linear(dim_mid, dim_out)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = getattr(self, f"layer_{i}")(x)
            x = self.relu(x)
        x = self.out(x)
        return self.softmax(x)


class VGGNet(nn.Module):
    """
    VGGNet Implementation with residual connections
    """

    def __init__(self):
        super().__init__()
        self.block1 = _conv_block(2, 3, 64, 3)
        self.block2 = _conv_block(2, 64, 128, 3)
        self.block3 = _conv_block(2, 128, 256, 3)
        self.block4 = _conv_block(3, 256, 512, 3)
        self.block5 = _conv_block(3, 512, 512, 3)
        self.flatten = nn.Flatten()
        self.fc = _fc_block(3, 25088, 100, 3)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.flatten(x)
        return self.fc(x)
