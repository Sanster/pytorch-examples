import math
import torch
from torch import nn
import torch.nn.functional as F


class ArcProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    """

    def __init__(self, in_features, out_features):
        super(ArcProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        return cosine


class ArcFaceLoss(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            embedding_dim: size of each input sample
            num_classes: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, embedding_dim=2, num_classes=10, s=7.0, m=0.30, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.in_features = embedding_dim
        self.out_features = num_classes
        self.s = s
        self.m = m
        self.fc = ArcProduct(embedding_dim, num_classes)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embedding, target):
        cosine = self.fc(embedding)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-8, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        x = F.log_softmax(output, dim=1)
        return F.nll_loss(x, target)
