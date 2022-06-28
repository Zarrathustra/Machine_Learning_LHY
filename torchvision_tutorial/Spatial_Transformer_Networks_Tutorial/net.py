import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Net']

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d( 1, 10, kernel_size = 5) # in-channel -> 1 ; out-channel -> 10, stride -> 1, kernel size -> 5
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5) # in-channel -> 10; out-channel -> 20, stride -> 1, kernel size -> 5

        # Convolutional Dropout
        self.conv2_drop = nn.Dropout2d()

        # Fully Connect Layer
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear( 50, 10)

        self.localization = nn.Sequential(nn.Conv2d(1, 8, kernel_size = 7), # out, 8 channels, N - kernel + 1 \
                                          nn.MaxPool2d(2, stride = 2),      # out, 8 channels, (N - kernel + 1) // 2 \
                                                                            #      downsample by factor 2 \
                                          nn.ReLU(inplace = True), \
                                          nn.Conv2d(8, 10, kernel_size = 5), # out, 10 channels \
                                          nn.MaxPool2d(2, stride=2),         #      dowsample by factor 2 \
                                          nn.ReLU(inplace = True))
        # output -> 10 channels, 3x3

        # input  -> 10 channels, 3x3 (output of self.localization)
        self.fc_loc = nn.Sequential(nn.Linear(10 * 3 * 3, 32), # flattened, then FC with 32 neurons \
                                    nn.ReLU(True), \
                                    nn.Linear(32, 3 * 2))
        # ouput -> affine transform, a 3x2 matrix -> affine transform, 6 parameters
        # Initialize the weights/bias with identity transformation
        # IMPORTANT FOR TRAINING
        # START WITH IDENTITY
        # GOOD START POINT
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def spatial_transform_network(self, x):

        xs = self.localization(x)
        # print(xs.view(-1, 10 * 3 * 3).shape) # size of each dimension, -1 stands for inferring from other dimensions

        theta = self.fc_loc(xs.view(-1, 10 * 3 * 3))

        # NOTE: self.localization + self.fc_loc -> a neural network for inferrring affine transforming parameters
        theta = theta.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x.size()) # get the irrgular grid
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):

        x = self.spatial_transform_network(x)

        # print(x.shape)

        # Perform the usual forward pass

        # Convolutional layer to 10 channels, then max pooling
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) 

        # Convolutional layer to 20 channels, then dropout, then max pooling
        # max pooling serves as a RELU in CNN
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # FC layers, 1
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))

        # FC layers, 2
        # dropout during training, not test and validation
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # output, 10 classes

        # a softmax for classification
        return F.log_softmax(x, dim = 1)

if __name__ == '__main__':

    x = torch.Tensor(10, 1, 28, 28) # 10 image, 1 channel, 28x28 -> MNIST DATASET
    # print(x)

    my_net = Net()

    # print(myNet)

    # print(my_net(x))
    print(my_net(x))
