# tools/convert_mat_to_npz.py
from pathlib import Path
import numpy as np
from scipy.io import loadmat

def _as1d(x):
    x = np.array(x)
    return x.reshape(-1) if (x.ndim == 2 and 1 in x.shape) else x

src = Path("data/point3.mat").resolve()
dst = Path("data/point3.npz").resolve()
print("Reading:", src)
mat = loadmat(src.as_posix())

point       = np.array(mat["point"], float)
d           = np.array(mat["d"], float)
bestroute   = _as1d(mat["bestroute"]).astype(int)
bestbreak   = _as1d(mat["bestbreak"]).astype(int)
time_windows= _as1d(mat["time_windows"]).astype(float)

np.savez(dst, point=point, d=d, bestroute=bestroute, bestbreak=bestbreak, time_windows=time_windows)
print("Saved:", dst)
