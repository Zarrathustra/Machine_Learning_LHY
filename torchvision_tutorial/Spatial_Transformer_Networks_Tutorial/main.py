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

# net
