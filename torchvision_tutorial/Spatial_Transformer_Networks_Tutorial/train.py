# License: BSD
# Author: Ghassen Hamrouni

from __future__ import print_function

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Torchvision
import torchvision
from torchvision import datasets, transforms

# Matplotlib
import matplotlib.pyplot as plt
plt.ion()   # interactive mode

# Numpy
import numpy as np

# Six
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data loader
from data_loader import train_loader, test_loader

# network
from net import Net

model = Net()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

def train(epoch):

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()

        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, \
                                                                           batch_idx * len(data), len(train_loader.dataset), \
                                                                           100. * batch_idx / len(train_loader), \
                                                                           loss.item()))
def test():

    with torch.no_grad():

        model.eval()
        test_loss = 0
        correct = 0

        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()

            # get the index of the max log-probability
            pred = output.max(1, keepdim = True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, \
                                                                                     correct, \
                                                                                     len(test_loader.dataset), \
                                                                                     100. * correct / len(test_loader.dataset)))
for epoch in range(1, 20 + 1):

    # epoch training
    train(epoch)

    # saving model
    torch.save(my_net.state_dict(), './model/model_epoch_{:3d}.ckpt'.format(epoch))

    # epoch test
    test()
