# tools/peek_mat.py
from pathlib import Path
import numpy as np
from scipy.io import loadmat

def _as1d(x):
    x = np.array(x)
    return x.reshape(-1) if (x.ndim == 2 and 1 in x.shape) else x

p = Path("data/point3.mat").resolve()
print("[MAT path]", p, "exists:", p.exists())

mat = loadmat(p.as_posix())
point = np.array(mat["point"], float)
bestroute = _as1d(mat["bestroute"]).astype(int)
bestbreak = _as1d(mat["bestbreak"]).astype(int)
time_windows = _as1d(mat["time_windows"]).astype(float)

print("point.shape       =", point.shape, " sample[0:3] =", point[:3])
print("bestroute.shape   =", bestroute.shape, " head =", bestroute[:5])
print("bestbreak.shape   =", bestbreak.shape, " =", bestbreak)
print("time_windows.size =", time_windows.size, " head =", time_windows[:5])

# Xuất CSV để soi ngang MATLAB openvar
np.savetxt("data/__point_from_mat.csv", point, delimiter=",")
np.savetxt("data/__bestroute_from_mat.csv", bestroute, fmt="%d", delimiter=",")
np.savetxt("data/__bestbreak_from_mat.csv", bestbreak, fmt="%d", delimiter=",")
np.savetxt("data/__time_windows_from_mat.csv", time_windows, delimiter=",")
print("Wrote CSVs under data/__*.csv")
