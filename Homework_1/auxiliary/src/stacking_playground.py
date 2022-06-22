import torch

import torch.nn as nn

class My_Model(nn.Module):

    def __init__(self, input_dim):

        super(My_Model, self).__init__()

        self.sub_module_A = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

        self.sub_module_B = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

        self.sub_module_C = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

        self.sub_module_D = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

        self.boss_module = nn.Sequential(
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):

        x_a = self.sub_module_A(x)
        x_b = self.sub_module_B(x)
        x_c = self.sub_module_C(x)
        x_d = self.sub_module_D(x)

        # print(x_a, x_b, x_c, x_d)
        # print(torch.cat((x_a, x_b, x_c, x_d), dim = 0))
        u = torch.cat((x_a, x_b, x_c, x_d), dim = 1)

        print(u.shape)

        u = self.boss_module(torch.cat((x_a, x_b, x_c, x_d), dim = 1))

        # print(u)

        # x = x.squeeze(1) # (B, 1) -> (B)
        u = u.squeeze(1) # (B, 1) -> (B)

        return u

if __name__ == '__main__':

    model = My_Model(24)

    x = torch.Tensor(256, 24)

    # print(x)

    print(model(x))
