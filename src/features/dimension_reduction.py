import numpy as np
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA

def fit_pca(X_train, X_test, X_val, n_components):
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
    assert mode in ['train', 'test',
                    'val'], "Mode must be either 'train', 'test', or 'val'"

    X = np.load(f'../data/processed/X_{mode}_embeddings.npy', mmap_mode='r')
    y = np.load(f'../data/processed/y_{mode}_labels.npy', mmap_mode='r')
    y = y.reshape(-1)
    return X, y
