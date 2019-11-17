import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, embedding_dim=2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, 1)
        self.conv2 = nn.Conv2d(8, 16, 5, 1)
        self.conv3 = nn.Conv2d(16, 8, 5, 1)
        self.fc1 = nn.Linear(128, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, 10)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, 1)
        fc1_out = self.prelu(self.fc1(x))
        x = self.fc2(fc1_out)
        return x, fc1_out


def softmax_loss(x, labels):
    x = F.log_softmax(x, dim=1)
    return F.nll_loss(x, labels)


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        embedding_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, embedding_dim=2):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.embedding_dim), requires_grad=True)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
