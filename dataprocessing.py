import numpy as np
from sklearn.decomposition import PCA
from mnist import MNIST

class DataProcessor:
    @staticmethod
    def reduce_dimension(X, n_components=2):
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        return X_reduced