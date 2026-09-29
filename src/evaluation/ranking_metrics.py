import numpy as np


def recall_at_k(scores, labels, k=5):
    if len(scores) == 0:
        return 0.0
    idx = np.argsort(np.asarray(scores))[::-1][:k]
    positives = labels[idx] == 1
    total = np.sum(labels == 1)
    return float(np.sum(positives) / max(total, 1))


def ndcg_at_k(scores, labels, k=5):
    if len(scores) == 0:
        return 0.0
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    idx = np.argsort(scores)[::-1][:k]
    gains = (2 ** labels[idx] - 1) / np.log2(np.arange(2, len(idx) + 2))
    ideal = np.sort(labels)[::-1][:k]
    ideal_gains = (2 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2))
    return float(np.sum(gains) / max(np.sum(ideal_gains), 1e-8))


def top_k_sensitivity(scores, labels, k=5):
    return recall_at_k(scores, labels, k)
