from torch import nn


class SoftmaxLoss(nn.Module):
    def __init__(self, embedding_dim=2, num_classes=10):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, embedding, labels):
        output = self.fc(embedding)
        return self.criterion(output, labels)
