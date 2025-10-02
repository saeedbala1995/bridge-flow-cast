import numpy as np
import joblib

def load_scaler_from_npz(npz_path):
    try:
        d = np.load(npz_path)
        if 'min_' in d and 'scale_' in d:
            class FakeScaler:
                def __init__(self, min_, scale_):
                    self.min_ = min_
                    self.scale_ = scale_
                def transform(self, X):
                    return (X - self.min_) / self.scale_
            return FakeScaler(d['min_'], d['scale_'])
    except Exception:
        pass
    return joblib.load(npz_path)

def ensure_3d_input(X):
    if X.ndim == 2:
        n, f = X.shape
        if f == 3:
            return X.reshape((n, 3, 1))
        return X.reshape((n, 1, f))
    return X
