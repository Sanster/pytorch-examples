import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from face_recognition_losses.model import Net, softmax_loss, CenterLoss

torch.manual_seed(42)


def train_epoch(args, model, device, train_loader, optimizer, epoch, center_loss=None, center_optimizer=None):
    model.train()

    embeddings = []
    labels = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        center_optimizer.zero_grad()
        output, embedding = model(data)
        # loss = softmax_loss(output, target)
        loss = softmax_loss(output, target) + center_loss(embedding, target)
        loss.backward()
        optimizer.step()
        center_optimizer.step()

        embeddings.append(embedding)
        labels.append(target)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    embeddings = torch.cat(embeddings, 0).detach().numpy()
    labels = torch.cat(labels, 0).detach().numpy()
    visualize(embeddings, labels, epoch)


def visualize(feat, labels, epoch):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.savefig('./images/epoch_%d.jpg' % epoch)


def val(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nTest set: Accuracy: {}/{} ({:.3f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=100)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(
        datasets.MNIST('../data', train=False, transform=transform),
        batch_size=512, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    center_loss = CenterLoss()
    center_optimizer = optim.Adadelta(center_loss.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_epoch(args, model, device, train_loader, optimizer, epoch,
                    center_loss=center_loss,
                    center_optimizer=center_optimizer)
        val(model, device, test_loader)


if __name__ == '__main__':
    main()
