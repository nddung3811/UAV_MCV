from __future__ import annotations
import os
import pandas as pd


def merge_instances(instances, results_root="results"):
    rows = []

    for inst in instances:
        path = os.path.join(
            results_root,
            f"instance_{inst}",
            "igd_vs_matlab_summary.csv"
        )

        if not os.path.isfile(path):
            print(f"[SKIP] {path} not found")
            continue

        df = pd.read_csv(path)
        rows.append(df.iloc[0].to_dict())

    if not rows:
        print("[ERROR] No IGD summary files found")
        return

    df_all = pd.DataFrame(rows)
    out_path = os.path.join(results_root, "igd_vs_matlab_all_instances.csv")
    df_all.to_csv(out_path, index=False)

    print(f"[OK] Saved {out_path}")


def main():
    merge_instances(instances=[1, 2, 3, 4, 5, 6])


if __name__ == "__main__":
    main()
