import numpy as np


def estimate_uncertainty(predictions):
    if len(predictions) == 0:
        return 0.0
    arr = np.asarray(predictions, dtype=np.float32)
    return float(np.mean(np.std(arr, axis=0)))
