# %%
# pytorch 1.0+
# modified from
# - https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/404_autoencoder.py
# - https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py
# - https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/variational_autoencoder/main.py
import argparse

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class VAE(nn.Module):
    # https://github.com/pytorch/examples/blob/master/vae/main.py
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class CnnAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),  # b, 16, 28, 28
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 14, 14
            nn.Conv2d(16, 32, 3, stride=1, padding=1),  # b, 32, 14, 14
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 32, 7, 7
            nn.Conv2d(32, 8, 3, stride=2, padding=1),  # b, 8, 4, 4
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 32, 3, stride=2),  # b, 32, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class LinearAutoEncoder(nn.Module):
    def __init__(self):
        super(LinearAutoEncoder, self).__init__()
        self.embedding_dim = 3
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, self.embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--model', default='cnn', choices=['cnn', 'linear', 'vae'])
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

    # Because ToTensor convert image in the range [0.0, 1.0], in the decoder last layer we should use sigmoid
    transform = transforms.Compose([transforms.ToTensor()])

    train_loader = DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    if args.model == 'linear':
        model = LinearAutoEncoder().to(device)
    elif args.model == 'cnn':
        model = CnnAutoEncoder().to(device)
    elif args.model == 'vae':
        model = VAE().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    frames = []
    if args.model == 'vae':
        fig, a = plt.subplots(3, 10, figsize=(10, 3))
    else:
        fig, a = plt.subplots(2, 10, figsize=(10, 2))

    # original data (first row) for viewing
    view_data = test_loader.dataset.data[:10].type(torch.FloatTensor).to(device) / 255.
    for i in range(10):
        a[0][i].imshow(np.reshape(view_data.cpu().data.numpy()[i], (28, 28)), cmap='gray')
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())
        a[1][i].set_xticks(())
        a[1][i].set_yticks(())
    fig.canvas.draw()  # draw the canvas, cache the renderer
    s, (width, height) = fig.canvas.print_to_buffer()
    image = np.fromstring(s, np.uint8).reshape((height, width, 4))
    frames.append(image)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            if args.model == 'vae':
                x_reconst, mu, logvar = model(data)
                loss = vae_loss_function(x_reconst, data, mu, logvar)
            else:
                encoded, decoded = model(data)
                loss = criterion(decoded.view(-1, 28 * 28), data.view(-1, 28 * 28))

            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

        if args.model == 'vae':
            decoded_data, _, _ = model(view_data.unsqueeze(1))
        else:
            _, decoded_data = model(view_data.unsqueeze(1))

        for i in range(10):
            a[1][i].clear()
            plt.title("epoch: %2d" % epoch)
            a[1][i].imshow(np.reshape(decoded_data.cpu().data.numpy()[i], (28, 28)), cmap='gray')
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())

            if args.model == 'vae':
                rand_in = torch.randn([1, 20]).to(device)
                rand_out = model.decode(rand_in)
                a[2][i].clear()
                a[2][i].imshow(np.reshape(rand_out.cpu().data.numpy(), (28, 28)), cmap='gray')
                a[2][i].set_xticks(())
                a[2][i].set_yticks(())

        fig.canvas.draw()  # draw the canvas, cache the renderer
        s, (width, height) = fig.canvas.print_to_buffer()
        image = np.fromstring(s, np.uint8).reshape((height, width, 4))
        frames.append(image)

    imageio.mimsave(f'{args.model}_autoencoder.gif', frames, 'GIF', duration=0.3)


if __name__ == '__main__':
    main()
