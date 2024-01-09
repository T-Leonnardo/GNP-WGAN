# -*-coding:utf-8-*-
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / (_range+1e-6)
def normalize(data, clp_s=3.2):
    z = (data - np.mean(data)) / np.std(data)
    return normalization(np.clip(z, a_min=-clp_s, a_max=clp_s))

def generate_mask(shape):
    a = int(random.randint(0,1))
    if a==0:
        mask = np.ones(shape)
        prop = random.randint(30, 90) / 100.
        y = random.sample(range(0, shape[1]), int(shape[1] * prop))
        for i in y:
            mask[:, i] = 0
    else:
        mask = np.ones(shape)
        cols = int(random.uniform(30, 100))#38,76,115
        start = random.randint(0, shape[1] - cols)
        mask[:, start:start + cols] = 0
    return mask

class MyDataset(Dataset):
    def __init__(self, feature_path,structure_paths):
        super(MyDataset, self).__init__()
        self.feature_paths = glob.glob(os.path.join(feature_path, '*.npy'))
        self.structure_paths = glob.glob(os.path.join(structure_paths, '*.npy'))

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, index):
        feature_data = np.load(self.feature_paths[index])
        feature_data = normalize(feature_data)
        structure_data = np.load(self.structure_paths[index])
        label = feature_data
        mask = generate_mask(feature_data.shape)
        feature = feature_data * mask
        structure_data = torch.from_numpy(structure_data)
        feature = torch.from_numpy(feature)
        label = torch.from_numpy(label)
        mask = torch.from_numpy(mask)
        structure_data.unsqueeze_(0)
        feature.unsqueeze_(0)
        mask.unsqueeze_(0)
        label.unsqueeze_(0)
        feature = feature+(1-mask)*structure_data
        return feature,label

    def visualize(self, index):
        feature, label= self.__getitem__(index)

        feature = feature.numpy().transpose(1, 2, 0)
        label = label.numpy().transpose(1, 2, 0)

        plt.subplot(1, 2, 1)
        plt.imshow(feature,cmap=plt.cm.seismic, interpolation='bilinear')
        plt.title("input")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(label, cmap=plt.cm.seismic, interpolation='bilinear')
        plt.title("Label")
        plt.axis('off')

        plt.tight_layout()
        plt.show()




