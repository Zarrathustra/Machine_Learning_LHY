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

# Mirror of PyTroch Dataset
# import torch_dataset_mirror
import torchvision.datasets.mnist as mnist

# Training dataset
train_loader = torch.utils.data.DataLoader(mnist.MNIST(root = '.', \
                                                       train = True, \
                                                       download = True, \
                                                       transform = transforms.Compose([transforms.ToTensor(), \
                                                                                       transforms.Normalize((0.1307,), (0.3081,))])), \
                                           batch_size = 64, \
                                           shuffle = True, \
                                           num_workers = 4)

"""
# Test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=4)
"""
