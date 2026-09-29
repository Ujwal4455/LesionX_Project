import numpy as np


def temperature_scale(logits, labels, temperature=1.0):
    logits = np.asarray(logits, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-logits / max(temperature, 1e-8)))


def expected_calibration_error(probs, labels, bins=10):
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for i in range(bins):
        a, b = edges[i], edges[i + 1]
        mask = (probs >= a) & (probs < b) if i < bins - 1 else (probs >= a) & (probs <= b)
        if not np.any(mask):
            continue
        avg_conf = np.mean(probs[mask])
        avg_acc = np.mean(labels[mask])
        score += (mask.mean()) * abs(avg_conf - avg_acc)
    return float(score)


def brier_score(probs, labels):
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    return float(np.mean((probs - labels) ** 2))
