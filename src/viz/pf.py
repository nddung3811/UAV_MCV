# src/viz/pf.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from ..metrics.topsis import topsis_matlab

def plot_pf(obj: np.ndarray,
            cost_divide_by: float = 5.0,
            title: str = "Pareto Front (cost vs deviation)",
            save: str | None = None):
    """
    Vẽ Pareto giống MATLAB:
    - obj: (n,2) [cost, deviation], đều là minimize
    - cost_divide_by: MATLAB hay vẽ obj(:,1)/5 để cân trục
    - Trả về (index_chosen, obj_chosen)
    """
    if obj.ndim != 2 or obj.shape[1] < 2:
        raise ValueError("obj must be a 2D array with at least 2 columns")

    # Chọn điểm theo TOPSIS (port từ MATLAB)
    obj_sel, index = topsis_matlab(obj)[:2]

    x = obj[:, 0] / float(cost_divide_by)
    y = obj[:, 1]

    fig, ax = plt.subplots(figsize=(7,5), dpi=120)
    ax.plot(x, y, "o", markersize=5, markeredgecolor="k",
            markerfacecolor=(0.608, 0.761, 0.902), label="PF points")
    ax.plot(obj[index, 0] / float(cost_divide_by), obj[index, 1], "o",
            markersize=8, markeredgecolor="k", markerfacecolor="r", label="TOPSIS best")

    ax.set_xlabel(f"Cost / {cost_divide_by}")
    ax.set_ylabel("Deviation")
    ax.set_title(title)
    ax.grid(True, alpha=.25)
    ax.legend(loc="best")
    if save:
        fig.savefig(save, bbox_inches="tight")
    plt.show()

    return int(index), obj_sel
