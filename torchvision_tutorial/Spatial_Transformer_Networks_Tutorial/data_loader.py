# License: BSD
# Author: Ghassen Hamrouni

__all__ = ['train_loader, test_loader']

import torch

# Mirror of PyTroch Dataset
# import torch_dataset_mirror
from torchvision import datasets, transforms
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

# Test dataset
test_loader = torch.utils.data.DataLoader(mnist.MNIST(root = '.', \
                                                      train = False, \
                                                      transform = transforms.Compose([transforms.ToTensor(), \
                                                                                      transforms.Normalize((0.1307,), (0.3081,))])), \
                                          batch_size = 64, \
                                          shuffle = True, \
                                          num_workers = 4)
