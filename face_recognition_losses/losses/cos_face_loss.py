# from https://github.com/MuggleWang/CosFace_pytorch
import torch
from torch import nn
import torch.nn.functional as F


def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1, w2).clamp(min=eps)


class CosineProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    """

    def __init__(self, in_features, out_features):
        super(CosineProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        cosine = cosine_sim(input, self.weight)
        return cosine


class CosFaceLoss(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        s: norm of input feature
        m: margin
    """

    def __init__(self, embedding_dim=2, num_classes=10, s=7, m=0.30):
        super(CosFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.fc = CosineProduct(embedding_dim, num_classes)

    def forward(self, embedding, target):
        cosine = self.fc(embedding)
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # --------------------------- convert label to one-hot ---------------------------
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (cosine - one_hot * self.m)

        x = F.log_softmax(output, dim=1)
        return F.nll_loss(x, target)
