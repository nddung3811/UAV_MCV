from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from src.analysis.export_csv import export_instance
from src.analysis.pareto import get_non_dominated_indices


def igd(pop_obj: np.ndarray, pf: np.ndarray) -> float:
    if pop_obj.size == 0 or pf.size == 0:
        return float("inf")
    D = cdist(pf, pop_obj)
    return float(D.min(axis=1).mean())


def export_igd(instance_id: int, results_root="results"):
    inst_dir = os.path.join(results_root, f"instance_{instance_id}")
    raw_csv = os.path.join(inst_dir, "raw_results.csv")

    if not os.path.isfile(raw_csv):
        print(f"[ERROR] raw_results.csv not found for instance {instance_id}")
        return

    df = pd.read_csv(raw_csv)

    # ==== build PF_ref (approximate true PF) ====
    all_pts = df[["cost", "tw_violation"]].values
    idx = get_non_dominated_indices(all_pts)
    pf_raw = all_pts[idx]

    # normalize
    mn = pf_raw.min(axis=0)
    mx = pf_raw.max(axis=0)
    rng = np.where(mx - mn == 0, 1.0, mx - mn)
    pf = (pf_raw - mn) / rng

    rows = []
    for seed in sorted(df["seed"].unique()):
        pts = df[df["seed"] == seed][["cost", "tw_violation"]].values
        pts_n = (pts - mn) / rng
        val = igd(pts_n, pf)
        rows.append({
            "instance": instance_id,
            "seed": seed,
            "igd": val
        })

    df_run = pd.DataFrame(rows)
    df_run.to_csv(os.path.join(inst_dir, "igd_per_run.csv"), index=False)

    summary = {
        "instance": instance_id,
        "best_igd": df_run["igd"].min(),
        "mean_igd": df_run["igd"].mean(),
        "std_igd": df_run["igd"].std(),
        "num_runs": len(df_run),
    }

    pd.DataFrame([summary]).to_csv(
        os.path.join(inst_dir, "igd_summary.csv"),
        index=False
    )

    print(f"[OK] IGD exported for instance {instance_id}")


def main():
    for inst in [1, 2, 3, 4, 5, 6]:
        export_igd(inst)


if __name__ == "__main__":
    main()
