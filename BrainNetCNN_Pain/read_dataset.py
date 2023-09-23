import numpy as np
import random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import os
from utils import matrix_preprocess, datasplit


class Dataset(Dataset):
    def __init__(self, root_dir, label_dir, dictionary, transform = transforms.ToTensor()):
        # data loading
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.filepath = os.path.join(self.root_dir,self.label_dir)
        self.txt_file = os.listdir(self.filepath)
        self.transform = transform
        self.dictionary = dictionary

    def __getitem__(self, index):
        # how to fetch the dataset
        txt_name = self.txt_file[index] #txt_name
        txt = os.path.join(self.root_dir, self.label_dir, txt_name) #txt_path
        data = np.loadtxt(txt)
        data = matrix_preprocess(data)
        label = self.dictionary[self.label_dir]

        if self.transform :
            data = self.transform(data)
        return data, label

    def __len__(self):
        # length of Dataset
        return len(self.txt_file)


categories = ['healthy', 'pain']
root_dir = 'D:\Pytorch\BrainNetCNN_pain//matrix108'

dict = {i:j for j, i in enumerate(categories)}
dataset = []

for i in range(len(categories)):
    label_dir = categories[i]
    x = Dataset(root_dir, label_dir,dictionary=dict)
    dataset.append(x)
full_dataset = ConcatDataset(dataset)
# healthy_dataset = Dataset(root_dir, 'healthy', dict)
# pain_dataset = Dataset (root_dir, 'pain', dict)
# full_dataset = ConcatDataset(healthy_dataset, pain_dataset)

train, test =  datasplit(full_dataset)
print(len(train),len(test))
test_loader = DataLoader(test)

# healthy = 0
# pain = 0
# for i, (data, target) in enumerate(train):
#     imgs, targets = train[i]
#     if targets == 0:
#         healthy +=1
#     elif targets == 1:
#         pain += 1
#     else:
#         continue
# print(healthy, pain)