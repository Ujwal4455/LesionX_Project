import numpy as np


def estimate_uncertainty(predictions):
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.size == 0:
        return 0.0
    return float(np.mean(np.std(predictions, axis=0)))
