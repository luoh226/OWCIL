import torch

from .ops import *
import torchvision.transforms as trn




class AvgOfPair(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shuffle_indices = np.arange(len(dataset))
        np.random.shuffle(self.shuffle_indices)

    def __getitem__(self, i):
        random_idx = np.random.choice(len(self.dataset))
        while random_idx == i:
            random_idx = np.random.choice(len(self.dataset))

        img1 = self.dataset[i][1]
        img1 = img1 / 2.
        img2 = self.dataset[random_idx][1]
        img2 = img2 / 2.
        return i, img1 + img2, -1

    def __len__(self):
        return len(self.dataset)


class GeomMeanOfPair(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shuffle_indices = np.arange(len(dataset))
        np.random.shuffle(self.shuffle_indices)

    def __getitem__(self, i):
        random_idx = np.random.choice(len(self.dataset))
        while random_idx == i:
            random_idx = np.random.choice(len(self.dataset))

        return i, torch.sqrt(self.dataset[i][1] * self.dataset[random_idx][1]), -1

    def __len__(self):
        return len(self.dataset)


class OodFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features):
        if not isinstance(features, torch.Tensor):
            features = torch.from_numpy(features)
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]

        return idx, feature, -1


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        # assert len(features) == len(labels), "Data size error!"
        if not isinstance(features, torch.Tensor):
            features = torch.from_numpy(features)
        self.features = features
        if not isinstance(labels, torch.Tensor):
            labels = torch.from_numpy(labels)
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return idx, feature, label