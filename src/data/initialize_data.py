# -*- coding: utf-8 -*-
import click
import logging

import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models

from torchvision import transforms

from src.data.preprocess_data import DatasetManager
from src.features.dimension_reduction import extract_cnn_features
from src.models.aws_utils import load_torch_model
import timeit

BATCH_SIZE = 8
INPUT_SIZE = 224
N_FEATURES = 7
VAL_SIZE = 0.2
TEST_SIZE = 0.2

DATA_PATH = 'data/processed'
RESNET_NAME = 'resnet-50.pt'
MODEL_PATH = 'models/'


def _create_cnn_embeddings(model):
    """
    Extract features, and save them locally. 
    This needs to be done only once, and later we can load the features from disk.

    Parameters
    ----------
    model : torchvision.models
        Must inherit torch.nn.Module. In other words must have forward().
    """
    print("Creating CNN embeddings..")
    start_t = timeit.default_timer()

    test_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(INPUT_SIZE),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_manager = DatasetManager(batch_size=BATCH_SIZE, test_size=TEST_SIZE, 
                        validation_size=VAL_SIZE, transform=test_preprocess, 
                        test_transform=test_preprocess)
    train_loader, test_loader, val_loader = dataset_manager.load_dataloaders()
    # Train
    embeddings, labels = extract_cnn_features(model, train_loader, BATCH_SIZE)
    np.save('../data/processed/y_train_labels_b7.npy', np.vstack(labels))
    np.save('../data/processed/X_train_embeddings_b7.npy', np.vstack(embeddings))
    # Test
    embeddings, labels = extract_cnn_features(model, test_loader, BATCH_SIZE)
    np.save('../data/processed/y_test_labels_b7.npy', np.vstack(labels))
    np.save('../data/processed/X_test_embeddings_b7.npy', np.vstack(embeddings))
    # Val
    embeddings, labels = extract_cnn_features(model, val_loader, BATCH_SIZE)
    np.save('../data/processed/y_val_labels_b7.npy', np.vstack(labels))
    np.save('../data/processed/X_val_embeddings_b7.npy', np.vstack(embeddings))

    print(f"CNN embedding saving done. Time taken: {timeit.default_timer() - start_t}s.")

def _create_model():
    """
    Create model that is used for generating embeddings.
    
    Returns
    -------
    resnet : torchvision.models.resnet50
    """
    resnet = models.resnet50(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(num_ftrs, 2048),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(1024,  N_FEATURES)
    )

    return resnet

def _load_checkpoint_to_local():
    """
    Load model checkpoint from S3, and save it to local directory.
    """
    print("Loading model checkpoint from S3 to local repository..")
    start_t = timeit.default_timer()
    checkpoint = load_torch_model(RESNET_NAME)
    print(f"Model checkpoint loaded in {timeit.default_timer() - start_t}")

    checkpoint['model_state_dict']
    torch.save(checkpoint, MODEL_PATH + RESNET_NAME)
    print("Model checkpoint saved!")


def main():
    """ 
    Generates CNN embeddings and loads model checkpoint from S3 to local.    
    """
    model = _create_model()
    _create_cnn_embeddings(model)
    _load_checkpoint_to_local()

    print("Data processing and loading has been done.")
    print("You can now start using the notebooks!")

if __name__ == '__main__':
    main()
