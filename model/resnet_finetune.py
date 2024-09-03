import torch.nn as nn
from torchvision import models

ResNet = models.resnet18(weights="DEFAULT")
num_classes = 3
in_features = ResNet.fc.in_features
ResNet.fc = nn.Linear(in_features, num_classes)
