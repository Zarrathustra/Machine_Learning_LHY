import pandas as pd

data = pd.read_csv('../data/covid.train.csv')

# print(data)

# print(data.columns)

pearson_corr = data.corr(method = 'pearson')

print(pearson_corr)
print(type(pearson_corr))
print(pearson_corr.iloc[-1])

c = pearson_corr.iloc[-1].abs()

print(c)

c = c.to_numpy()

print(c)

print(c > 0.8)

print(data.columns)

print(data.columns[c > 0.8])

print(data.columns[c > 0.8].shape)

"""
import numpy as np

print(np.where(c > 0.8))
# print(pearson_corr[-1, :])
"""

import torch

a = torch.Tensor(3)
print(a)

b = torch.Tensor(3)
print(b)

c = torch.cat((a, b), dim = 0)
print(c)
