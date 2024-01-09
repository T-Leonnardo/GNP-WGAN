# -*-coding:utf-8-*-
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random
from models.NLE import get_NLE

# normalize
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
        cols = int(random.uniform(30, 100))
        start = random.randint(0, shape[1] - cols)
        mask[:, start:start + cols] = 0
    return mask

class MyDataset(Dataset):
    def __init__(self, feature_path):
        super(MyDataset, self).__init__()
        self.feature_paths = glob.glob(os.path.join(feature_path, '*.npy'))

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, index):
        data = np.load(self.feature_paths[index])
        data = normalize(data)
        feature_data = get_NLE(data)
        label_data = feature_data
        mask = generate_mask(data.shape)
        feature_data = feature_data * mask
        mask = torch.from_numpy(mask)
        feature_data.unsqueeze_(0)
        label_data.unsqueeze_(0)
        mask.unsqueeze_(0)
        # feature_data = torch.concat((feature_data,mask),dim=0)
        return feature_data,mask,label_data

    def visualize(self, index):
        feature, mask, label = self.__getitem__(index)

        feature = feature.numpy().transpose(1, 2, 0)
        label = label.numpy().transpose(1, 2, 0)
        mask = mask.numpy().transpose(1, 2, 0)

        plt.subplot(1, 3, 1)
        plt.imshow(feature, cmap="gray", interpolation='bilinear',aspect=1,vmin=0,vmax=1)
        plt.title("Feature")

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap="gray", interpolation='bilinear',aspect=1,vmin=0,vmax=1)
        plt.title("mask")

        plt.subplot(1, 3, 3)
        plt.imshow(label, cmap="gray", interpolation='bilinear',aspect=1,vmin=0,vmax=1)
        plt.title("Label")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    feature_path = "../data/data_C3/features"
    seismic_dataset = MyDataset(feature_path)
    train_size = int(0.8 * len(seismic_dataset))
    test_size = len(seismic_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(seismic_dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=1,
                                               shuffle=False)
    print('train_data size:', len(train_dataset))
    print('train_loader:', len(train_loader))
    print('val_dataset size:', len(seismic_dataset))
    print('val_loader:', len(val_loader))
    seismic_dataset.visualize(1)