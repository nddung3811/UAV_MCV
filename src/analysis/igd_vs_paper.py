from __future__ import annotations
import os
import numpy as np
from scipy.spatial.distance import cdist
from scipy.io import loadmat


def igd(pop_obj: np.ndarray, pf: np.ndarray) -> float:
    """
    IGD kiểu PlatEMO:
      Distance = min(pdist2(PF,PopObj),[],2);
      Score    = mean(Distance);
    pf:  (K, M)  - PF_ref đã normalize
    pop_obj: (N, M) - front của thuật toán đã normalize
    """
    if pop_obj.size == 0 or pf.size == 0:
        return float("inf")
    D = cdist(pf, pop_obj)
    dmin = D.min(axis=1)
    return float(dmin.mean())


def load_matlab_refs(tpf_path: str, norm_path: str):
    """
    Đọc PF_ref (all_obj) và min1/max1 từ MATLAB.
    Trả về:
        all_obj_list: list length 6, mỗi phần tử là (K_j, 2)
        min1: (6, 2)
        max1: (6, 2)
    """
    tpf = loadmat(tpf_path)        # chứa all_obj (1x6 cell)
    norm = loadmat(norm_path)      # chứa min1, max1

    # all_obj là cell 1x6, loadmat -> mảng (1,6) dtype=object
    all_obj_cells = tpf["all_obj"][0]
    all_obj_list = [np.asarray(all_obj_cells[j], dtype=float) for j in range(6)]

    min1 = np.asarray(norm["min1"], dtype=float)  # (6, 2)
    max1 = np.asarray(norm["max1"], dtype=float)  # (6, 2)
    return all_obj_list, min1, max1


def load_my_runs(result_root: str, inst_id: int):
    """
    Đọc tất cả run_*.npz của GA-OBL cho một instance.
    Mỗi file có 'pareto_obj' (K,2) = [cost, TW_violation].
    """
    inst_dir = os.path.join(result_root, f"instance_{inst_id}")
    if not os.path.isdir(inst_dir):
        return []

    runs = []
    files = []
    for fname in sorted(os.listdir(inst_dir)):
        if not fname.endswith(".npz"):
            continue
        path = os.path.join(inst_dir, fname)
        data = np.load(path)
        if "pareto_obj" not in data:
            continue
        runs.append(np.asarray(data["pareto_obj"], dtype=float))
        files.append(fname)
    return runs, files


def main():
    # Đường dẫn đến file .mat của MATLAB
    TPF_PATH   = "data/TPF.mat"        # sửa theo repo của ông
    NORM_PATH  = "data/normalize.mat"  # sửa đường dẫn nếu cần
    RESULT_DIR = "results"              # nơi GA-OBL lưu run_XXX.npz

    # 1) Load PF_ref + min/max từ MATLAB
    all_obj_list, min1, max1 = load_matlab_refs(TPF_PATH, NORM_PATH)

    print("Inst |  best_IGD |  mean_IGD |   std_IGD | runs (GA-OBL vs PF_ref MATLAB)")
    print("---------------------------------------------------------------")

    for inst in range(1, 7):
        # 2) PF_ref cho instance này (RAW)
        pf_ref_raw = all_obj_list[inst - 1]    # (K,2)
        if pf_ref_raw.size == 0:
            print(f"{inst:4d} |        - |        - |        - | 0 (no PF_ref)")
            continue

        # 3) Min/max cho instance này (giống normalize.mat)
        mn = min1[inst - 1, :]     # (2,)
        mx = max1[inst - 1, :]     # (2,)
        rng = np.where(mx - mn == 0.0, 1.0, mx - mn)

        # Normalize PF_ref
        pf_ref = (pf_ref_raw - mn[None, :]) / rng[None, :]

        # 4) Load tất cả run của GA-OBL cho instance này
        runs, files = load_my_runs(RESULT_DIR, inst)
        if not runs:
            print(f"{inst:4d} |        - |        - |        - | 0 (no runs)")
            continue

        igd_vals = []
        for fname, pop_raw in zip(files, runs):
            pop_n = (pop_raw - mn[None, :]) / rng[None, :]
            val = igd(pop_n, pf_ref)
            igd_vals.append(val)
            print(f"    [Inst {inst}] {fname}: IGD = {val:.4f}")

        igd_arr = np.asarray(igd_vals, dtype=float)
        best = float(igd_arr.min())
        mean = float(igd_arr.mean())
        std  = float(igd_arr.std())

        print(f"{inst:4d} | {best:9.4f} | {mean:9.4f} | {std:9.4f} | {len(igd_arr)}")

if __name__ == "__main__":
    main()
