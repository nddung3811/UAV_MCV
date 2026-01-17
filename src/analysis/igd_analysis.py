from __future__ import annotations
import os
import numpy as np
from scipy.spatial.distance import cdist
from .pareto import get_non_dominated_indices


def igd(pop_obj: np.ndarray, pf: np.ndarray) -> float:
    """
    IGD đúng kiểu PlatEMO:
        Distance = min(pdist2(PF,PopObj),[],2);
        Score    = mean(Distance);
    Ở đây:
        pf      ~ PF (reference front), shape (K, M)
        pop_obj ~ PopObj (front của 1 run), shape (N, M)
    """
    if pop_obj.size == 0 or pf.size == 0:
        return float("inf")
    D = cdist(pf, pop_obj)      # (K, N)
    dmin = D.min(axis=1)        # (K,)
    return float(dmin.mean())


def _load_runs(dir_path: str) -> list[np.ndarray]:
    """Đọc tất cả run_*.npz trong 1 thư mục instance."""
    runs = []
    if not os.path.isdir(dir_path):
        return runs
    for f in sorted(os.listdir(dir_path)):
        if not f.endswith(".npz"):
            continue
        data = np.load(os.path.join(dir_path, f))
        if "pareto_obj" not in data:
            continue
        runs.append(data["pareto_obj"].astype(float))
    return runs


def analyze_instance(dir_path: str):
    """
    - Gom tất cả điểm Pareto của mọi run -> union
    - Lọc non-dominated -> PF_ref (approx true PF)
    - Chuẩn hoá PF_ref và từng run theo min/max của PF_ref
    - Tính IGD cho từng run, rồi trả về (best, mean, std, số run)
    """
    runs = _load_runs(dir_path)
    if not runs:
        return None, None, None, 0

    # ==== build PF_ref: union + non-dominated ====
    all_points = np.vstack(runs)                # (Tổng_K, 2)
    idx = get_non_dominated_indices(all_points)
    pf_ref_raw = all_points[idx]               # (K, 2)

    # ==== normalize theo min/max (min1, max1) ====
    min1 = pf_ref_raw.min(axis=0)
    max1 = pf_ref_raw.max(axis=0)
    rng = np.where(max1 - min1 == 0.0, 1.0, max1 - min1)

    pf_ref = (pf_ref_raw - min1) / rng         # PF_ref chuẩn hoá

    igds = []
    for f, pareto in zip(sorted(os.listdir(dir_path)), runs):
        pop_n = (pareto - min1) / rng          # front của run chuẩn hoá
        val = igd(pop_n, pf_ref)
        igds.append(val)
        print(f"    {f}: IGD = {val:.4f}")

    igds = np.asarray(igds, dtype=float)
    return igds.min(), igds.mean(), igds.std(), len(igds)


def main(base_dir: str = "results"):
    """
    base_dir:
        - Nếu bạn đang lưu ở results/instance_1/... thì để mặc định.
        - Nếu bạn lưu ở notebooks/instance_1/... thì gọi:
              python -m src.uav_mcv.analyze_igd notebooks
    """
    print("Inst |  best_IGD |  mean_IGD |   std_IGD | runs")
    print("-----------------------------------------------")

    for inst in range(1, 7):
        dir_path = os.path.join(base_dir, f"instance_{inst}")
        if not os.path.exists(dir_path):
            print(f"{inst:4d} |        - |        - |        - | 0")
            continue

        best, mean, std, runs = analyze_instance(dir_path)
        print(f"{inst:4d} | {best:9.4f} | {mean:9.4f} | {std:9.4f} | {runs}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
