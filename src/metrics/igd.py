from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist

def igd(pop_obj: np.ndarray, pf: np.ndarray) -> float:
    """
    Inverted Generational Distance (IGD)
    - pop_obj: quần thể (n, m)
    - pf: Pareto front tham chiếu (k, m)
    Trả về IGD = mean( min_{x in pop} dist(pf_i, x) )
    """
    if pop_obj.ndim != 2 or pf.ndim != 2:
        raise ValueError("pop_obj and pf must be 2D arrays")
    # Khoảng cách từ mỗi điểm pf tới tập pop_obj
    D = cdist(pf, pop_obj)
    dmin = D.min(axis=1)
    return float(dmin.mean())
