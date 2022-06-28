from PIL import Image

import requests

# PyTorch
import torch

# Torchvision
import torchvision
import torchvision.transforms as transforms

# PyTorch Image Model
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# loading Pre-trained model
model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained = True)
model.eval()

transform = transforms.Compose([transforms.Resize(256, interpolation=3), \
                                transforms.CenterCrop(224), \
                                transforms.ToTensor(), \
                                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])


if __name__ == '__main__':

    img = Image.open(requests.get("https://raw.githubusercontent.com/pytorch/ios-demo-app/master/HelloWorld/HelloWorld/HelloWorld/image.png", stream = True).raw)

    print(img)

    print(torch.__version__)
