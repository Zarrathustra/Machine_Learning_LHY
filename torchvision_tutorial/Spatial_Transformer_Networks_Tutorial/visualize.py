# PyTorch
import torch

# Torchvision
import torchvision

# Matplotlib
import matplotlib.pyplot as plt

# Numpy
import numpy as np

# device setting
from train import device

def convert_image_np(inp):

    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    return inp

# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.

def visualize_stn():

    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()

        transformed_input_tensor = model.spatial_transform_network(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

if __name__ == '__main__':

    from net import Net

    model = Net()

    model.load_state_dict(torch.load('./model/model_epoch_020.ckpt'))

    from data_loader import test_loader

    # Visualize the STN transformation on some input batch
    visualize_stn()

    plt.savefig('image.png', dpi = 200)
