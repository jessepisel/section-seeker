import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

"""
Collection of SNN compatible wrappers for general MNIST-like dataset
"""


class ContrastiveDataset(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    
    :param dataset: MNIST-like  VisionDataset to get pair from
    """

    def __init__(self, dataset):
        
        self.dataset = dataset

        self.train = self.dataset.train
        self.transform = self.dataset.transform
        self.colormode = self.dataset.colormode
        self.compatibility_mode = self.dataset.compatibility_mode
        
        if self.train:
            self.train_labels = self.dataset.targets
            self.train_data = self.dataset.data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.dataset.targets
            self.test_data = self.dataset.data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}
            
            random_state = np.random.RandomState(29)
                
            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]
        
        if self.compatibility_mode:
            img1 = Image.fromarray(img1.numpy(), mode=self.colormode)
            img2 = Image.fromarray(img2.numpy(), mode=self.colormode)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
                
        img1 = img1.transpose(2,0).transpose(2,1)
        img2 = img2.transpose(2,0).transpose(2,1)
        return (img1, img2), target

    def __len__(self):
        return len(self.dataset)


class TripletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.train = self.dataset.train
        self.transform = self.dataset.transform
        self.colormode = self.dataset.colormode
        self.compatibility_mode = self.dataset.compatibility_mode 
        
        if self.train:
            self.train_labels = self.dataset.targets
            self.train_data = self.dataset.data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.dataset.targets
            self.test_data = self.dataset.data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]
        
        if self.compatibility_mode:
            img1 = Image.fromarray(img1.numpy(), mode=self.colormode)
            img2 = Image.fromarray(img2.numpy(), mode=self.colormode)
            img3 = Image.fromarray(img3.numpy(), mode=self.colormode)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            
        img1 = img1.transpose(2,0).transpose(2,1)
        img2 = img2.transpose(2,0).transpose(2,1)
        img3 = img3.transpose(2,0).transpose(2,1)
        
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.dataset)

