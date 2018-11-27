import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np


class CelebDataset(Dataset):
    def __init__(self, image_path, metadata_path, transform, mode, crop_size):
        self.image_path = image_path
        self.transform = transform
        self.mode = mode
        self.lines = open(metadata_path, 'r').readlines()
        # self.num_data = int(self.lines[0])
        self.num_data = len(self.lines)
        self.crop_size = crop_size

        print ('Start preprocessing dataset..!')
        random.seed(1234)
        self.preprocess()
        print ('Finished preprocessing dataset..!')

        if self.mode == 'train':
            self.num_data = len(self.train_filenames)
        elif self.mode == 'test':
            self.num_data = len(self.test_filenames)

    def preprocess(self):
        self.train_filenames = []
        self.test_filenames = []

        lines = self.lines[2:]
        random.shuffle(lines)   # random shuffling
        for i, line in enumerate(lines):

            splits = line.split()
            filename = splits[0]

            if (i+1) < 20000:
                self.test_filenames.append(filename)
            else:
                self.train_filenames.append(filename)

    def __getitem__(self, index):
        if self.mode == 'train':
            image = Image.open(os.path.join(self.image_path, self.train_filenames[index]))
        elif self.mode in ['test']:
            image = Image.open(os.path.join(self.image_path, self.test_filenames[index]))
        # self.check_size(image, index)
        return self.transform(image)

    def __len__(self):
        return self.num_data

class PlacesDataset(Dataset):
    def __init__(self, image_path, metadata_path, transform, mode, crop_size):
        self.image_path = image_path
        self.transform = transform
        self.mode = mode
        self.lines = open(metadata_path, 'r').readlines()
        self.num_data = len(self.lines)
        self.crop_size = crop_size

        print ('Start preprocessing dataset..!')
        random.seed(1234)
        self.preprocess()
        print ('Finished preprocessing dataset..!')

        if self.mode == 'train':
            self.num_data = len(self.train_filenames)
            print('training len:'+str(self.num_data))
        elif self.mode == 'test':
            self.num_data = len(self.test_filenames)
            print('testing len:'+str(self.num_data))

    def preprocess(self):
        self.train_filenames = []
        self.test_filenames = []

        lines = self.lines[0:200000]
        print(len(lines))
        # random.shuffle(lines)   # random shuffling # deleteeeeeee
        for i, line in enumerate(lines):

            splits = line.split()
            filename = splits[0]

            if (i) < 20000:
                self.test_filenames.append(filename)
            else:
                self.train_filenames.append(filename)

    def __getitem__(self, index):
        if self.mode == 'train':
            image = Image.open(os.path.join(self.image_path, self.train_filenames[index])).convert('RGB')
            # print(self.train_filenames[index])
            # image_ = np.array(image)
            # print(image_.shape)
        elif self.mode in ['test']:
            image = Image.open(os.path.join(self.image_path, self.test_filenames[index]))
        # self.check_size(image, index)
        return self.transform(image)

    def __len__(self):
        return self.num_data

def get_loader(image_path, metadata_path, crop_size, image_size, batch_size, dataset='Places2', mode='train'):
    """Build and return data loader."""

    if mode == 'train':
        transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Resize(image_size, interpolation=Image.ANTIALIAS),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Scale(image_size, interpolation=Image.ANTIALIAS),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    if dataset == 'CelebA':
        dataset = CelebDataset(image_path, metadata_path, transform, mode, crop_size)
    elif dataset == 'Places2':
        dataset = PlacesDataset(image_path, metadata_path, transform, mode, crop_size)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader