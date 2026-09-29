import numpy as np
from sklearn.metrics.pairwise import cosine_distances


class KNNOutlierDetector:
    def __init__(self, k=5, metric="cosine"):
        self.k = k
        self.metric = metric
        self.embeddings = None

    def fit(self, X):
        self.embeddings = np.asarray(X, dtype=np.float32)
        return self

    def score(self, X):
        if self.embeddings is None:
            raise ValueError("Call fit() before score().")
        X = np.asarray(X, dtype=np.float32)
        dist = cosine_distances(X, self.embeddings)
        k = min(self.k, dist.shape[1])
        knn = np.partition(dist, kth=k - 1, axis=1)[:, :k]
        return knn.mean(axis=1)
