from __future__ import annotations
import os
import numpy as np
import pandas as pd


def export_instance(instance_id: int, results_root="results"):
    inst_dir = os.path.join(results_root, f"instance_{instance_id}")

    if not os.path.isdir(inst_dir):
        print(f"[SKIP] {inst_dir} not found")
        return

    rows = []

    for fname in sorted(os.listdir(inst_dir)):
        if not fname.endswith(".npz"):
            continue

        seed = int(fname.split("_")[1].split(".")[0])
        path = os.path.join(inst_dir, fname)
        data = np.load(path)

        if "pareto_obj" not in data:
            continue

        objs = data["pareto_obj"]
        for i, (cost, tw) in enumerate(objs):
            rows.append({
                "instance": instance_id,
                "seed": seed,
                "solution_id": i,
                "cost": float(cost),
                "tw_violation": float(tw),
            })

    if not rows:
        print(f"[WARN] No data found in {inst_dir}")
        return

    df = pd.DataFrame(rows)
    out_path = os.path.join(inst_dir, "raw_results.csv")
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved {out_path}")


def main():
    # sửa list này theo instance m có
    for inst in [1, 2, 3, 4, 5,6]:
        export_instance(inst)


if __name__ == "__main__":
    main()
