from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from ..decision.topsis import topsis_matlab


def plot_pareto_front(
    objectives: np.ndarray,
    cost_scale: float = 5.0,
    title: str = "Pareto Front (cost vs deviation)",
    save_path: str | None = None,
) -> tuple[int, np.ndarray]:
    """
    Plot Pareto front for a bi-objective minimization problem.

    Parameters
    ----------
    objectives : (N, 2) ndarray
        Objective values [cost, deviation].
    cost_scale : float
        Scaling factor applied to cost for visualization.
    title : str
        Plot title.
    save_path : str | None
        If provided, save figure to this path.

    Returns
    -------
    index_best : int
        Index of the solution selected by TOPSIS.
    obj_best : ndarray
        Objective values of the selected solution.
    """
    if objectives.ndim != 2 or objectives.shape[1] < 2:
        raise ValueError("objectives must have shape (N, 2) or more columns")

    # Select representative solution using TOPSIS
    obj_best, index_best = topsis_matlab(objectives)[:2]

    x = objectives[:, 0] / float(cost_scale)
    y = objectives[:, 1]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)

    ax.plot(
        x, y, "o",
        markersize=5,
        markeredgecolor="k",
        markerfacecolor=(0.608, 0.761, 0.902),
        label="Pareto solutions",
    )

    ax.plot(
        objectives[index_best, 0] / float(cost_scale),
        objectives[index_best, 1],
        "o",
        markersize=8,
        markeredgecolor="k",
        markerfacecolor="r",
        label="TOPSIS selection",
    )

    ax.set_xlabel(f"Cost / {cost_scale}")
    ax.set_ylabel("Deviation")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    plt.show()

    return int(index_best), obj_best
