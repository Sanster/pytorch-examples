# from https://github.com/KaiyangZhou/pytorch-center-loss
import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    """

    def __init__(self, embedding_dim=2, num_classes=10, cent_w=1):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.cent_w = cent_w
        # In origin paper, self.centers are updated by delta of centers computed use embedding and self.centers
        # In this code, self.centers are updated by the backpropagation of the loss.
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.embedding_dim), requires_grad=True)

        self.fc = nn.Linear(embedding_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, embedding, labels):
        batch_size = embedding.size(0)
        distmat = torch.pow(embedding, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, embedding, self.centers.t())

        classes = torch.arange(self.num_classes).long().cuda()
        mask = labels.unsqueeze(1).expand(batch_size, self.num_classes).eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        x = self.fc(embedding)
        return self.criterion(x, labels) + self.cent_w * loss
