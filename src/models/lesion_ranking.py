import numpy as np


def recall_at_k(scores, labels, k=5):
    if len(scores) == 0:
        return 0.0
    idx = np.argsort(scores)[::-1][:k]
    total_pos = int(np.sum(labels == 1))
    if total_pos == 0:
        return 0.0
    return float(np.sum(labels[idx] == 1) / total_pos)


def ndcg_at_k(scores, labels, k=5):
    if len(scores) == 0:
        return 0.0
    idx = np.argsort(scores)[::-1][:k]
    sorted_labels = np.asarray(labels, dtype=float)
    gains = (2 ** sorted_labels[idx] - 1) / np.log2(np.arange(2, len(idx) + 2))
    ideal = np.sort(sorted_labels)[::-1][:k]
    ideal_gains = (2 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2))
    return float(np.sum(gains) / max(np.sum(ideal_gains), 1e-8))


def top_k_sensitivity(scores, labels, k=5):
    return recall_at_k(scores, labels, k)
