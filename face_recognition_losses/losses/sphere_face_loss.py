# From https://github.com/clcarwin/sphereface_pytorch/
# https://www.cnblogs.com/darkknightzh/p/8524937.html
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def myphi(x, m):
    x = x * m
    return 1 - x ** 2 / math.factorial(2) + x ** 4 / math.factorial(4) - x ** 6 / math.factorial(6) + \
           x ** 8 / math.factorial(8) - x ** 9 / math.factorial(9)


torch.autograd.set_detect_anomaly(True)


class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.phiflag = phiflag

        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input):
        # size=(B, embedding_dim)
        x = input

        weight_normed = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5)  # size=B
        wlen = weight_normed.pow(2).sum(0).pow(0.5)  # size=Classnum

        cos_theta = x.mm(weight_normed) / (xlen.view(-1, 1) * wlen.view(1, -1))
        cos_theta = cos_theta.clamp(-1, 1)

        if self.phiflag:
            theta = cos_theta.data.acos()
            k = (self.m * theta / 3.14159265).floor()
            n_one = k * 0.0 - 1

            cos_m_theta = self.mlambda[self.m](cos_theta)
            phi_theta = (n_one ** k) * cos_m_theta - 2 * k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta, self.m)
            phi_theta = phi_theta.clamp(-1 * self.m, 1)

        cos_theta = cos_theta * xlen.view(-1, 1)
        phi_theta = phi_theta * xlen.view(-1, 1)
        output = (cos_theta, phi_theta)
        return output  # size=(B,Classnum,2)


class SphereFaceLoss(nn.Module):
    def __init__(self, embedding_dim=2, num_classes=10, gamma=0, m=4):
        super(SphereFaceLoss, self).__init__()
        self.gamma = gamma
        self.iter = 0
        self.lambda_min = 5.0
        self.lambda_max = 1500.0
        self.lamb = 1500.0
        self.fc = AngleLinear(embedding_dim, num_classes, m=m)

    def forward(self, embedding, target):
        input = self.fc(embedding)
        self.iter += 1
        cos_theta, phi_theta = input
        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)

        self.lamb = max(self.lambda_min, self.lambda_max / (1 + 0.1 * self.iter))
        output = cos_theta.clone()
        output = output - cos_theta * index / (1 + self.lamb)
        output = output + phi_theta * index / (1 + self.lamb)

        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = loss.mean()
        return loss
