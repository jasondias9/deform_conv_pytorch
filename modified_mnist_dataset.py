import errno
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch

class ModMNISTDataset(Dataset):
    
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root_dir, train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train = train
        self.root = root_dir
        self.transform = transform
        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.training_file))
            self.train_data = self.train_data / 255.
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.test_file))
            self.test_data = self.test_data / 255.
                
    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))
        
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(img.numpy()*255), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target 