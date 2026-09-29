import numpy as np


def fuse_features(visual=None, metadata=None, morphology=None, tbp=None):
    parts = []

    if visual is not None:
        parts.append(np.asarray(visual, dtype=np.float32))
    if metadata is not None:
        parts.append(np.asarray(metadata, dtype=np.float32))
    if morphology is not None:
        parts.append(np.asarray(morphology, dtype=np.float32))
    if tbp is not None:
        parts.append(np.asarray(tbp, dtype=np.float32))

    if not parts:
        raise ValueError("No feature tensors were provided.")

    fused = parts[0]
    for p in parts[1:]:
        fused = np.concatenate([fused, p], axis=-1)
    return fused.astype(np.float32)
