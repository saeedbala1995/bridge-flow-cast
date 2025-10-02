import numpy as np
import joblib
import os

def load_scaler(path):
    """
    Load a fitted scaler from file.
    Supports:
      - .npz files saved with numpy (must contain min_ and scale_)
      - .joblib files saved with sklearn's joblib
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler file not found: {path}")

    # Try numpy .npz format
    if path.endswith(".npz"):
        d = np.load(path)
        if "min_" in d and "scale_" in d:
            class FakeScaler:
                def __init__(self, min_, scale_):
                    self.min_ = min_
                    self.scale_ = scale_
                def transform(self, X):
                    return (X - self.min_) / self.scale_
            return FakeScaler(d["min_"], d["scale_"])

    # Try joblib format
    try:
        return joblib.load(path)
    except Exception as e:
        raise ValueError(
            f"Could not load scaler from {path}. "
            "Supported formats: .npz (with min_/scale_) or joblib."
        ) from e


def ensure_3d_input(X):
    """
    Ensure input array has 3 dimensions (samples, timesteps, features).
    If X has shape (n,3), reshape to (n,3,1).
    """
    if X.ndim == 2:
        n, f = X.shape
        if f == 3:
            return X.reshape((n, 3, 1))
        return X.reshape((n, 1, f))
    return X
