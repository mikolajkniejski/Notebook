# Basen on https://neptune.ai/blog/gumbel-softmax-loss-function-guide-how-to-implement-it-in-pytorch

# %%
import numpy as np
from typing import Union, Optional, List, Tuple, Text, BinaryIO
import io
import pathlib
import math
irange = range

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

# %%
batch_size = 100
epochs = 10
temperature = 1.0
no_cuda = False
seed = 2020
log_interval = 10
hard = False # Nature of Gumbel-softmax

# %%

is_cuda = not no_cuda and torch.cuda.is_available()
torch.manual_seed(seed)
if is_cuda:
torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}

train_loader = torch.utils.data.DataLoader(
datasets.MNIST('./data/MNIST', train=True, download=True,
transform=transforms.ToTensor()),
batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
datasets.MNIST('./data/MNIST', train=False, transform=transforms.ToTensor()),
batch_size=batch_size, shuffle=True, **kwargs)

# %%

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    if is_cuda:
        U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)
def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)
def gumbel_softmax(logits, temperature, hard=False):

    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y.view(-1, latent_dim * categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim * categorical_dim)

# %%

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False) / x.shape[0]

    log_ratio = torch.log(qy * categorical_dim + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).mean()

    return BCE + KLD

class VAE_gumbel(nn.Module):
    def __init__(self, temp):
        super(VAE_gumbel, self).__init__()

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim * categorical_dim)

        self.fc4 = nn.Linear(latent_dim * categorical_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def decode(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.sigmoid(self.fc6(h5))

    def forward(self, x, temp, hard):
        q = self.encode(x.view(-1, 784))
        q_y = q.view(q.size(0), latent_dim, categorical_dim)
        z = gumbel_softmax(q_y, temp, hard)
        return self.decode(z), F.softmax(q_y, dim=-1).reshape(*q.size())

# %%
latent_dim = 30
categorical_dim = 10  # one-of-K vector

temp_min = 0.5

ANNEAL_RATE = 0.00003

model = VAE_gumbel(temperature)

if is_cuda:
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# %%


def train(epoch):
    model.train()
    train_loss = 0
    temp = temperature
    for batch_idx, (data, _) in enumerate(train_loader):
        if is_cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, qy = model(data, temp, hard)
        loss = loss_function(recon_batch, data, qy)
        loss.backward()
        train_loss += loss.item() * len(data)
        optimizer.step()
        if batch_idx % 100 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)

        if batch_idx==0:
            reconstructed_image = recon_batch.view(batch_size, 1, 28, 28)
            grid_array = get_grid(reconstructed_image)

            run["train_reconstructed_images/{}".format('training_reconstruction_' + str(epoch))].upload(File.as_image(grid_array))
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    run['metrics/avg_train_loss'].log(train_loss / len(train_loader.dataset))
def test(epoch):
    model.eval()
    test_loss = 0
    temp = temperature
    for i, (data, _) in enumerate(test_loader):
        if is_cuda:
            data = data.cuda()
        recon_batch, qy = model(data, temp, hard)
        test_loss += loss_function(recon_batch, data, qy).item() * len(data)
        if i % 100 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * i), temp_min)
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(batch_size, 1, 28, 28)[:n]])
            grid_array = get_grid(comparison)

            run["test_reconstructed_images/{}".format('test_reconstruction_' + str(epoch))].upload(File.as_image(grid_array))

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    run['metrics/avg_test_loss'].log(test_loss)


# %%

for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)


