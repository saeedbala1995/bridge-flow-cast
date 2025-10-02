from utils import ensure_3d_input
import numpy as np

def test_ensure_3d_input():
    X = np.array([[1,2,3],[4,5,6]])
    X3 = ensure_3d_input(X)
    assert X3.shape == (2,3,1)
