from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt


# ======================================================
# Non-dominated filtering (minimize both objectives)
# ======================================================
def get_non_dominated(points: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if (
                np.all(points[j] <= points[i])
                and np.any(points[j] < points[i])
            ):
                keep[i] = False
                break
    return keep


# ======================================================
# Plot settings (white background – paper friendly)
# ======================================================
def _set_white_style():
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "grid.alpha": 0.3,
        "font.size": 11,
    })


# ======================================================
# Main function
# ======================================================
def plot_instance(instance_dir: str, save_dir: str):
    instance_name = os.path.basename(instance_dir)

    # ---------- load all runs ----------
    all_points = []
    run1_points = None

    for fname in sorted(os.listdir(instance_dir)):
        if not fname.endswith(".npz"):
            continue

        data = np.load(os.path.join(instance_dir, fname))
        if "pareto_obj" not in data:
            continue

        pts = data["pareto_obj"][:, :2]
        all_points.append(pts)

        if fname == "run_001.npz":
            run1_points = pts

    if not all_points:
        print(f"[WARN] {instance_name}: no data")
        return

    all_points = np.vstack(all_points)

    # ---------- approximate PF ----------
    mask = get_non_dominated(all_points)
    pf = all_points[mask]
    pf = pf[np.argsort(pf[:, 0])]  # sort by cost

    # ---------- plot ----------
    _set_white_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    # all points (faint)
    ax.scatter(
        all_points[:, 0],
        all_points[:, 1],
        s=20,
        alpha=0.35,
        color="#9ecae1",
        label="All solutions",
    )

    # run 1
    if run1_points is not None:
        ax.scatter(
            run1_points[:, 0],
            run1_points[:, 1],
            s=35,
            color="#3182bd",
            edgecolors="k",
            linewidths=0.5,
            label="Run 1 Pareto",
        )

    # approximate PF
    ax.plot(
        pf[:, 0],
        pf[:, 1],
        "k-",
        linewidth=2.2,
        label="Approximate Pareto front",
    )

    ax.set_xlabel("Cost")
    ax.set_ylabel("Time window violation")
    ax.set_title(f"Pareto Front – {instance_name}", weight="bold")
    ax.grid(True, linestyle="--")
    ax.legend()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{instance_name}_pareto.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved {save_path}")


# ======================================================
# Run for all instances
# ======================================================
def main():
    RESULT_ROOT = "results"
    SAVE_ROOT = "results/pareto_plots"

    for name in sorted(os.listdir(RESULT_ROOT)):
        inst_dir = os.path.join(RESULT_ROOT, name)
        if not os.path.isdir(inst_dir):
            continue
        if not name.startswith("instance_"):
            continue

        plot_instance(inst_dir, SAVE_ROOT)


if __name__ == "__main__":
    main()
