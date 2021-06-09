from typing import List
import io
import os

import torch
from torch.utils.data import Subset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from PIL import Image

import numpy as np
import pandas as pd

# S3 bucket
import boto3

from s3fs.core import S3FileSystem

EMOTION_LIST = ['Angry', 'Disgust', 'Fear',
                'Happy', 'Sad', 'Surprise', 'Neutral']
N_FEATURES = len(EMOTION_LIST)


class FerDataset(Dataset):
    """
    Holds FER dataset.  

    Parameters
    ----------
    X : array-like of shape (n_samples, img_height, img_width)
        Predictor variables.

    Y : array-like of shape (n_samples, n_categories)
        Target variables.

    gray_to_rgb : bool, default=True
        Whether or not to convert predictor array X 
        to 3 channel image from 1 channel.

    transform : bool, default=None
        Whether or not to apply transformation on predictor variables.

    """

    def __init__(self, X, y, gray_to_rgb=True, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.gray_to_rgb = gray_to_rgb

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx]

        if self.gray_to_rgb:
            X = np.stack((X.squeeze(),)*3, axis=-1)

        if self.transform:
            X = self.transform(X)

        y = self.y[idx]

        return X, y


class Print(torch.nn.Module):
    """ Useful for debugging intermediate shapes in Compose. """

    def __init__(self):
        super().__init__()

    def forward(self, img):
        print(f"img: {img.shape}")
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


def _index_to_emotion(index):
    """Convert index to emotion"""
    return EMOTION_LIST[index]


class DatasetManager():
    """
    This is a class for generating dataloaders, and managing data-dependant operations.

    Parameters
    ----------
    batch_size : int

    test_size: float, default=0.2
        Controls the testset size taken from all data.

    validation_size: float, default=0.2
            Controls the validation set size taken from all data.

    transform: torchvision.transforms.Compose, default=None
        Preprocessing applied to training dataset.

    test_transform: torchvision.transforms.Compose, default=None 
        Preprocessing applied to training dataset.
    """
    def __init__(self, batch_size, test_size=0.2,
                validation_size=0.2,
                transform=None, test_transform=None):
        self.batch_size = batch_size
        self.test_size = test_size
        self.validation_size = validation_size
        self.transform = transform
        self.test_transform = test_transform

    def load_array_from_s3(self, category: int) -> np.ndarray:
        """
        Loads numpy array from S3 bucket for a given category.

        Returns
        -------

        """
        s3fs = S3FileSystem()
        key = f'data/{category}.npy'
        bucket = 'loopqprize'
        arr = np.load(s3fs.open('{}/{}'.format(bucket, key)))
        return arr

    def split_data(self, arr, shuffle=True, random_state=1):
        """
        Splits data to train/test/val

        Returns
        -------
        trainset : np.ndarray

        testset : np.ndarray

        valset : np.ndarray
        """
        trainset, testset = train_test_split(
            arr, test_size=self.test_size, shuffle=shuffle, random_state=random_state)
        trainset, valset = train_test_split(
            trainset, test_size=self.validation_size, shuffle=shuffle, random_state=random_state)

        return trainset, testset, valset


    def load_dataloaders(self, return_raw_data=False):
        '''
        Creates and returns DataLoaders.

        Returns
        -------
        train_loader : DataLoader
            Dataloader for training set.

        val_loader : DataLoader
            Dataloader for validation set.

        test_loader : DataLoader
            Dataloader for test set.
        '''
        s3 = boto3.resource('s3')

        train_X_arrays = []
        train_y_arrays = []
        val_X_arrays = []
        val_y_arrays = []
        test_X_arrays = []
        test_y_arrays = []

        # Iterate through every emotion, and split to train/test/val
        for i, emotion in enumerate(EMOTION_LIST):
            # Load from S3
            arr = self.load_array_from_s3(i)
            train_set, test_set, val_set = self.split_data(arr)
            
            y_train = np.ones(len(train_set)) * i
            y_test = np.ones(len(test_set)) * i
            y_val = np.ones(len(val_set)) * i

            train_X_arrays.append(train_set)
            train_y_arrays.append(y_train)
            test_X_arrays.append(test_set)
            test_y_arrays.append(y_test)
            val_X_arrays.append(val_set)
            val_y_arrays.append(y_val)

        # Concatenate to single array
        X_train = np.concatenate(train_X_arrays)
        y_train = np.concatenate(train_y_arrays)
        X_test = np.concatenate(test_X_arrays)
        y_test = np.concatenate(test_y_arrays)
        X_val = np.concatenate(val_X_arrays)
        y_val = np.concatenate(val_y_arrays)

        if return_raw_data:
            return {'X': {'train': X_train, 'test': X_test, 'val': X_val},
                    'y': {'train': y_train, 'test': y_test, 'val': y_val}}

        # Create datasets
        train_dataset = FerDataset(X_train, y_train, transform=self.transform)
        test_dataset = FerDataset(X_test, y_test, transform=self.test_transform)
        val_dataset = FerDataset(X_val, y_val, transform=self.transform)
        
        # Save for later usage
        self.y_train = y_train

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                shuffle=True, pin_memory=True)

        return train_loader, test_loader, val_loader

    def calculate_class_weights(self):
        """ 
        Calculate class weights proportional to class imbalances.

        Returns
        -------
        class_weights : ndarray of shape (n_classes, )
            Array with class_weights[i] the weight for i-th class.
        """
        class_weights = compute_class_weight(
            'balanced', np.unique(self.y_train), self.y_train)
        return class_weights
