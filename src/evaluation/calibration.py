import numpy as np


def temperature_scale(logits, labels, temperature=1.0):
    logits = np.asarray(logits, dtype=np.float32)
    probs = 1.0 / (1.0 + np.exp(-logits / max(temperature, 1e-8)))
    return probs


def expected_calibration_error(probs, labels, bins=10):
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        left, right = edges[i], edges[i + 1]
        mask = (probs >= left) & (probs <= right)
        if i < bins - 1:
            mask = (probs >= left) & (probs < right)
        if not np.any(mask):
            continue
        avg_conf = np.mean(probs[mask])
        avg_acc = np.mean(labels[mask])
        ece += (mask.sum() / len(labels)) * abs(avg_conf - avg_acc)
    return float(ece)


def brier_score(probs, labels):
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    return float(np.mean((probs - labels) ** 2))
