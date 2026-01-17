from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.io import loadmat


def igd(pop_obj: np.ndarray, pf: np.ndarray) -> float:
    D = cdist(pf, pop_obj)
    return float(D.min(axis=1).mean())


def export_igd_matlab(
    instance_id: int,
    tpf_path="data/TPF.mat",
    norm_path="data/normalize.mat",
    results_root="results",
):
    # ===== load MATLAB reference =====
    tpf = loadmat(tpf_path)
    norm = loadmat(norm_path)

    pf_ref_raw = np.asarray(tpf["all_obj"][0][instance_id - 1], dtype=float)
    min1 = np.asarray(norm["min1"][instance_id - 1], dtype=float)
    max1 = np.asarray(norm["max1"][instance_id - 1], dtype=float)
    rng = np.where(max1 - min1 == 0, 1.0, max1 - min1)

    pf_ref = (pf_ref_raw - min1) / rng

    # ===== load GA-OBL runs =====
    inst_dir = os.path.join(results_root, f"instance_{instance_id}")
    rows = []

    for fname in sorted(os.listdir(inst_dir)):
        if not fname.endswith(".npz"):
            continue
        seed = int(fname.split("_")[1].split(".")[0])
        data = np.load(os.path.join(inst_dir, fname))

        pop = np.asarray(data["pareto_obj"], dtype=float)
        pop_n = (pop - min1) / rng

        val = igd(pop_n, pf_ref)
        rows.append({
            "instance": instance_id,
            "seed": seed,
            "igd": val
        })

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(inst_dir, "igd_vs_matlab_per_run.csv"),
        index=False
    )

    summary = {
        "instance": instance_id,
        "best_igd": df["igd"].min(),
        "mean_igd": df["igd"].mean(),
        "std_igd": df["igd"].std(),
        "num_runs": len(df),
    }

    pd.DataFrame([summary]).to_csv(
        os.path.join(inst_dir, "igd_vs_matlab_summary.csv"),
        index=False
    )

    print(f"[OK] IGD vs MATLAB exported for instance {instance_id}")


def main():
    for inst in [1, 2, 3, 4, 5, 6]:
        export_igd_matlab(inst)


if __name__ == "__main__":
    main()
