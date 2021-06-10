import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm


def fit_pca(X_train, X_test, X_val, n_components):
    """
    Fits Principal Component Analysis on training data (first parameter),
    and transforms training, validation and test data accordingly. 

    Parameters
    ----------
    X_train : np.ndarray
        PCA will be fit on this array. 

    X_test : np.ndarray
        Test data that will be transformed.

    X_val : np.ndarray
        Validation data that will be transformed.

    n_components : int
        The amount of principal components PCA should produce


    Returns
    -------
    pca : sklearn.PCA
        Fitten PCA algorithm.

    X_train: np.ndarray
        Transformer training data.

    X_test: np.ndarray
        Transformed testing data.

    X_val: np.ndarray
        Transformed validation data.
    """

    n_rows_train = X_train.shape[0]
    n_rows_test = X_test.shape[0]
    n_rows_val = X_val.shape[0]

    X_train_flatten = X_train.reshape(n_rows_train, -1)
    X_val_flatten = X_val.reshape(n_rows_val, -1)
    X_test_flatten = X_test.reshape(n_rows_test, -1)

    pca = PCA(n_components=n_components, whiten=True).fit(X_train_flatten)
    # Apply transformation
    X_train = pca.transform(X_train_flatten)
    X_test = pca.transform(X_test_flatten)
    X_val = pca.transform(X_val_flatten)

    return pca, X_train, X_test, X_val


def extract_cnn_features(model, loader, batch_size):
    """
    Generate embeddings from last layer of the model.

    Parameters
    ----------
    model : torch.model
        Model which will be used for generating embeddings

    loader : DataLoader
        DataLoader where data is stored, and embeddings are created for.

    batch_size : int
        Batch size for the DataLoader.

    Returns
    ------- 
    embeddings : List[np.ndarray] of shape (n_batches, batch_size, n_features_flattened)
        Returns the feature representations as flattened arrays. For example EfficientNet-b0 returns 1280x7x7 embeddings,
        so the resulting n_features_flattened
    """

    embeddings = []
    labels = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for X, y in tqdm(loader):
            X, y = X.to(device), y
            if y.shape[0] == batch_size:
                features = model.extract_features(X)
                # Flatten features
                feat_flat = features.detach().flatten(1).cpu().numpy()

                labels.append(y.numpy())
                embeddings.append(feat_flat)

    return embeddings, labels

def load_cnn_embedding(mode: str):
    """
    Loads CNN embedding from the local storage.
    NOTE: Before this the local folders must be populated with embedding data.
    This can be done with the script `initialize_project.sh`.
    
    Parameters
    ----------
    mode : str, in ['train', 'test', 'val']
        Which CNN embedding should be loaded.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_embeddings) 
        Embeddings of the predictor variables

    y : np.ndarray of shape (n_samples, )
        The target values, stored just for consistency purposes.
    """

    assert mode in ['train', 'test',
                    'val'], "Mode must be either 'train', 'test', or 'val'"

    X = np.load(f'../data/processed/X_{mode}_embeddings.npy', mmap_mode='r')
    y = np.load(f'../data/processed/y_{mode}_labels.npy', mmap_mode='r')
    y = y.reshape(-1)
    return X, y
