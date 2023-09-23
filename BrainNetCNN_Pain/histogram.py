import matplotlib.pyplot as plt
import numpy as np
import torch

import scipy
import scipy.interpolate

from read_dataset import test, train
from utils import data_preprocess, random_datasplit

#load gpu model
PATH = 'D:\Pytorch\model\BrainNetCNN_pain\gpu\E2E_512_b4_k3_r0_leaky_e150_0.94_0.79.pt'
model = torch.load(PATH)

# PATH =
# train_features, train_labels = next(iter(train_dataloader))
# model = BrainNetCNN(train_features)
# model.load_state_dict(torch.load(PATH))
#

def plot_histogram (data, model, iteration=200,bins=5):
    accuracy_matrix = []
    for i in range(1, iteration):
        accuracy = 0
        train, test = random_datasplit(data)
        for input, target in train:
            input = data_preprocess(input)
            output = model(input)
            _, pred = torch.max(output, dim=1)
            pred_label = pred.cpu().numpy()[0]
            acc = int(target == pred_label)
            accuracy += acc
        accuracy = accuracy / len(train)
        accuracy_matrix.append(accuracy)

    n,binEdges,_ = plt.hist(accuracy_matrix,bins=bins,alpha = 0.4, label = 'Histogram')
    bin_centers = 0.5*(binEdges[1:]+binEdges[:-1])
    center_space = bin_centers[1]-bin_centers[0]
    new_center = np.insert(bin_centers,0,bin_centers[0]-center_space)
    new_center = np.append(new_center,bin_centers[-1]+center_space)
    f = scipy.interpolate.interp1d(bin_centers,n,fill_value='extrapolate')
    added_n = f(new_center)
    plt.plot(new_center,added_n,'--',label = 'Predictive fitted Curve')
    plt.plot(bin_centers,n,'-',label = 'Fitted Curve' )
    plt.xlim(bin_centers[0]-2*center_space,bin_centers[-1]+2*center_space)
    plt.ylim(np.max(bin_centers))
    plt.title('Histogram')
    plt.xlabel('Accuracy')
    plt.ylabel('Occurrence')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_histogram(test, model,200)
