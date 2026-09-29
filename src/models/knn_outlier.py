import numpy as np
from sklearn.metrics.pairwise import cosine_distances


class KNNOutlierDetector:
    def __init__(self, k=5, metric="cosine"):
        self.k = k
        self.metric = metric
        self.embedding_matrix = None

    def fit(self, X):
        self.embedding_matrix = np.asarray(X, dtype=np.float32)
        return self

    def score(self, X):
        if self.embedding_matrix is None:
            raise ValueError("Call fit() before score().")
        X = np.asarray(X, dtype=np.float32)
        dist = cosine_distances(X, self.embedding_matrix)
        topk = np.partition(dist, kth=min(self.k, dist.shape[1] - 1), axis=1)[:, : self.k]
        return np.mean(topk, axis=1)
