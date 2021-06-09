import os
import torch

from s3fs.core import S3FileSystem
from joblib import load

MODEL_PATH = 'models'
CNN_FEATURE_PATH = 'cnn_features'
PCA_FEATURE_PATH = 'pca_features'
HOG_FEATURE_PATH = 'hog_features'
BUCKET = 'loopqprize'

def load_torch_model(model_name: str):
    """
    Loads pre-trained torch model checkpoint from S3 bucket.

    Parameters
    ----------
    model_name : str
        Specify name of the pre-trained torch model checkpoint.

    Returns
    -------
    checkpoint : dict
        Dictionary as checkpoint for saved torch model
    """
    s3fs = S3FileSystem()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(s3fs.open(f'{BUCKET}/{MODEL_PATH}/{model_name}'),
                        map_location=device)

    return checkpoint

def load_traditional_model(model_type: str):
    model_type = model_type.lower()
    assert model_type in ['cnn', 'pca', 'hog']

    s3fs = S3FileSystem()
    if model_type == 'cnn':
        model_dir = f"{BUCKET}/{MODEL_PATH}/{CNN_FEATURE_PATH}"
    elif model_type == 'pca':
        model_dir = f"{BUCKET}/{MODEL_PATH}/{PCA_FEATURE_PATH}"
    elif model_type == 'hog':
        model_dir = f"{BUCKET}/{MODEL_PATH}/{HOG_FEATURE_PATH}"

    models = [(load(s3fs.open(item['name'])), item['name'].split('/')[-1])
              for item in s3fs.listdir(model_dir)]

    return models
