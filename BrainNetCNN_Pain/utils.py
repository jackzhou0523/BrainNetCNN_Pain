import numpy as np
import torch
import torch.utils.data
import math
import matplotlib.pyplot as plt


# 1) remove NaNs
# 2) convert Inf into 0
def matrix_preprocess(a):
    x = a[:,~np.all(np.isnan(a), axis=0)]
    mask = np.all(np.isnan(x),axis = 1)
    x = x[~mask]
    x[x == np.inf] = 0
    return x

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()


def init_weights_he(net, m):
    #https://keras.io/initializers/#he_uniform
    print(m)
    if type(m) == torch.nn.Linear:
        fan_in = net.dense1.in_features
        he_lim = np.sqrt(6) / fan_in
        m.weight.data.uniform_(-he_lim,he_lim)


def datasplit(dataset, rate=0.8, seed = 0):
    train_size = int(rate * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                [train_size, test_size],
                                                                generator=torch.Generator().manual_seed(seed))

    return train_dataset, test_dataset

def random_datasplit(dataset, rate=0.8):
    train_size = int(math.ceil(rate * len(dataset)))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                [train_size, test_size])
    return train_dataset, test_dataset


def data_preprocess(input,device="cuda:0"):
    input= input.to(device=device, dtype=torch.float)
    input = input.unsqueeze(0)
    return input

def normalize(slc, threshold=0.0):
    slc = (slc - slc.min()) / (slc.max() - slc.min())
    slc[slc < threshold] = 0
    k = slc
    return k

def plot2(a,b,threshold=0.0):
    x = a-b
    x = normalize(x, threshold)
    plot(x)

def plot(a, threshold=0.0):
    a = normalize(a, threshold)
    plt.figure(figsize=(10, 10))
    plt.imshow(a, cmap=plt.cm.jet)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.show()




