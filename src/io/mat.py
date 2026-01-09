from __future__ import annotations
from scipy.io import loadmat, savemat


def load_mat(path: str) -> dict:
    """
    Load .mat file and remove MATLAB internal fields.
    """
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def save_mat(path: str, **data):
    """
    Save variables to a .mat file.
    """
    savemat(path, data)
