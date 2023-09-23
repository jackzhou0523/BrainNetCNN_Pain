import numpy as np
import torch

from read_dataset import test
from saliency_map import saliency
from utils import plot, normalize

# cpu model load
model = torch.load('D:\Pytorch\model\BrainNetCNN_pain\cpu\E2E_b4_k5_r0_test_tanh_e150_0.88_0.96.pt')

# PATH = 'D:\Pytorch\model\dict//brainNetCnn//0904//3fc_46_256_128_32_2_500'
# device = torch.device('cpu')
# train_features, train_labels = next(iter(train_dataloader))
# model_2 = BrainNetCNN(train_features)
# model_2.load_state_dict(torch.load(PATH, map_location=device))

# heatmap generation
# dictionary = {'carriers':0, 'healthy':1, 'patients':2}
healthy  = np.zeros((108, 108))
count_healthy = 0
pain = np.zeros((108, 108))
count_pain = 0
count_wrong = 0


for data, target in test:
    slc, pred= saliency(data, model)
    if target == pred == 0:
        healthy += slc
        count_healthy += 1
    elif target == pred == 1:
        pain += slc
        count_pain += 1

    else:
        count_wrong +=1

#plot heatmap
plot(healthy)
plot(pain)

# plot2(healthy,patients)
# x = healthy-patients


# save成edge檔案 供matlab使用
# def save_edge(x, fname):
#     mat = np.matrix(x)
#     filename = fname + '.edge'
#     with open(filename,'wb') as f:
#         for line in mat:
#             np.savetxt(f, line, fmt='%.12f')
#
# filename = 'c-h'
# filepath = 'C://Users\Jack Zhou\Desktop//'+ filename
# save_edge(normalize(carriers-healthy), filepath)
