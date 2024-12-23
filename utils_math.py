import numpy as np

def inv_lerp(v, a, b):
    return (v-a)/(b-a)


def reject_outliers(data:np.ndarray, m:float = 2.0) -> np.ndarray:
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]