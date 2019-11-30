import argparse
import os
from pathlib import Path

import torch
import imageio
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
# from model import ConvNet as Net
from model import Net
from losses import CenterLoss, softmax_loss, SphereFaceLoss, CosFaceLoss, SoftmaxLoss, ArcFaceLoss

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def visualize(viz, loss_name, feat, labels, epoch, acc):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    out_dir = os.path.join('./%s' % viz, loss_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    plt.title("epoch: %2d   accuracy: %.4f" % (epoch, acc))
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, 'epoch_{:02d}.jpg'.format(epoch)))


def create_gif(save_path, jpgs_dir, duration=0.2):
    frames = []
    if not Path(save_path).parent.exists():
        os.makedirs(Path(save_path).parent)
    image_list = list(Path(jpgs_dir).glob('*.jpg'))
    image_list.sort()
    for image_name in image_list:
        frames.append(imageio.imread(image_name))

    # make last frame show longer
    for _ in range(5):
        frames.append(imageio.imread(image_list[-1]))
    imageio.mimsave(save_path, frames, 'GIF', duration=duration)


def val(model, criterion, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            embedding = model(data)
            output = criterion.fc(embedding)

            if isinstance(output, tuple):
                output = output[0]

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    acc = correct / len(test_loader.dataset) * 100
    print('\nTest set: Accuracy: {}/{} ({:.3f}%)\n'.format(
        correct, len(test_loader.dataset), acc))
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--loss', default='softmax_loss',
                        choices=['softmax_loss', 'center_loss', 'sphere_face_loss', 'cos_face_loss', 'arc_face_loss'])
    parser.add_argument('--viz', default='vizs')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

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

    if args.loss == 'center_loss':
        criterion = CenterLoss().to(device)
        center_optimizer = optim.SGD([criterion.centers], lr=args.lr, momentum=0.9)
    elif args.loss == 'sphere_face_loss':
        criterion = SphereFaceLoss().to(device)
    elif args.loss == 'cos_face_loss':
        criterion = CosFaceLoss(s=7, m=0.2).to(device)
    elif args.loss == 'softmax_loss':
        criterion = SoftmaxLoss().to(device)
    elif args.loss == 'arc_face_loss':
        criterion = ArcFaceLoss().to(device)

    optimizer = optim.SGD([{'params': model.parameters()}, {'params': criterion.fc.parameters()}],
                          lr=args.lr, momentum=0.9)

    for epoch in range(1, args.epochs + 1):
        model.train()
        embeddings = []
        labels = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            if args.loss == 'center_loss':
                center_optimizer.zero_grad()
            embedding = model(data)

            loss = criterion(embedding, target)
            loss.backward()
            optimizer.step()
            if args.loss == 'center_loss':
                center_optimizer.step()

            embeddings.append(embedding)
            labels.append(target)

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

        embeddings = torch.cat(embeddings, 0).cpu().detach().numpy()
        labels = torch.cat(labels, 0).cpu().detach().numpy()
        acc = val(model, criterion, device, test_loader)
        visualize(args.viz, args.loss, embeddings, labels, epoch, acc)

    print('Creating gif...')
    create_gif('./%s/gifs/%s.gif' % (args.viz, args.loss),
               './%s/%s' % (args.viz, args.loss), 0.2)
    print('Done')


if __name__ == '__main__':
    main()
