import torch
import torch.nn as nn
from torchvision import models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        embedding_dim = 2
        self.res18 = models.resnet18()
        self.res18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.res18.fc = nn.Linear(512, embedding_dim)

    def forward(self, x):
        embedding = self.res18(x)
        return embedding
