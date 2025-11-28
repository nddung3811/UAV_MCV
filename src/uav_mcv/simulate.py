from __future__ import annotations
import numpy as np
from .. import config
from ..io.mat import load_mat
from ..moea.selection import tournament_selection, environmental_selection
from ..moea.ga import GA
from ..moea.obl import obl
from ..metrics.topsis import topsis_matlab
from ..viz.routes import plot_mcv_route, plot_uav_route
from .initialize import initialize
from .fitness import cal_obj_con, CalFitness

# =============== Utilities ===============
def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))

def _build_route(bestroute: np.ndarray, bestbreak: np.ndarray, nn: int) -> np.ndarray:
    """
    Đúng logic MATLAB:
      break1 = [0, bestbreak, len(bestroute)];
      route  = [nn, bestroute(bi+1:bi+1), nn, ...]
    Giữ chỉ số 1-based giống MATLAB.
    """
    br = np.atleast_1d(bestbreak).astype(int).ravel()
    K  = int(np.atleast_1d(bestroute).size)
    # luôn chèn 0 ở đầu và K ở cuối như MATLAB
    br_full = np.concatenate(([0], br, [K]))
    rt = [nn]
    for i in range(len(br_full) - 1):
        a = br_full[i] + 1          # MATLAB 1-based
        b = br_full[i + 1]
        if a <= b:
            seg = np.atleast_1d(bestroute)[a - 1:b]  # numpy 0-based slice
            rt.extend(seg.tolist())
        rt.append(nn)
    return np.array(rt, dtype=int)

def _bounds_for_edges(route: np.ndarray,
                      point: np.ndarray,
                      start_point: np.ndarray,
                      n_vehicle: int,
                      t_char_min: float,
                      t_char_max: float,
                      scope: float = 6.0):
    """
    lower/upper cho bộ biến (dài 3*n_route):
      - Block 1 (0..n_route-1): ID MCV ∈ [0..n_vehicle]
      - Block 2 (n_route..2*n_route-1): tỷ lệ sạc ∈ [lo_i..hi_i]
      - Block 3 (2*n_route..3*n_route-1): thời gian sạc ∈ [t_char_min..t_char_max]
    route 1-based; depot ảo là nn = N+1.
    """
    n_route = len(route) - 1
    nn = point.shape[0] + 1

    lower = np.zeros(3 * n_route, dtype=float)
    upper = np.zeros(3 * n_route, dtype=float)

    # Block 1: MCV id
    lower[:n_route] = 0.0
    upper[:n_route] = float(n_vehicle)

    # Block 2: charge ratio per edge
    for i in range(n_route):
        u, v = int(route[i]), int(route[i + 1])
        if u == nn:
            d1 = _euclid(point[v - 1], start_point)
            lo = 0.0
            hi = (d1 - scope) / d1 if d1 > 0 else 0.0
        elif v == nn:
            d1 = _euclid(point[u - 1], start_point)
            lo = (scope / d1) if d1 > 0 else 0.0
            hi = 1.0
        else:
            d1 = _euclid(point[u - 1], point[v - 1])
            lo = (scope / d1) if d1 > 0 else 0.0
            hi = (d1 - scope) / d1 if d1 > 0 else 0.0
        lower[n_route + i] = max(0.0, min(1.0, lo))
        upper[n_route + i] = max(0.0, min(1.0, hi))

    # Block 3: charge time
    lower[2 * n_route:3 * n_route] = float(t_char_min)
    upper[2 * n_route:3 * n_route] = float(t_char_max)
    return lower, upper

# =============== Main pipeline ===============
def main(seed: int | None = None):
    # Lấy seed: ưu tiên tham số, nếu None thì lấy từ config.SEED (mặc định 42)
    if seed is None:
        seed = getattr(config, "SEED", 42)

    # Có thể seed luôn global RNG nếu còn chỗ nào lỡ dùng np.random
    np.random.seed(seed)
    # RNG chuẩn để truyền xuống các hàm
    rng = np.random.default_rng(seed)

    start_point = np.array(config.START_POINT, dtype=float)
    n_vehicle  = config.N_VEHICLE
    t_char_max = 30.0
    t_char_min = 5.0
    p_obl      = 0.8
    pop_size   = 200
    gen        = 200

    # === chọn instance từ config nếu có, mặc định point3.mat ===
    instance = getattr(config, "INSTANCE", "data/point3.mat")
    data = load_mat(instance)
    point        = np.array(data["point"], dtype=float)
    bestroute    = np.array(data["bestroute"], dtype=int).ravel()
    bestbreak    = np.array(data["bestbreak"], dtype=int).ravel()
    time_windows = np.array(data["time_windows"]) if "time_windows" in data else None

    nn = point.shape[0] + 1
    route = _build_route(bestroute, bestbreak, nn)   # 1-based
    n_route = len(route) - 1

    lower, upper = _bounds_for_edges(route, point, start_point,
                                     n_vehicle, t_char_min, t_char_max, scope=6.0)

    # Khởi tạo quần thể bằng rng
    pop1 = initialize(pop_size, n_route, lower, upper, rng=rng)
    pop2 = initialize(pop_size, n_route, lower, upper, rng=rng)

    obj1, con1, *_ = cal_obj_con(pop1, route, n_route, pop_size, point, time_windows)
    obj2, con2, *_ = cal_obj_con(pop2, route, n_route, pop_size, point, time_windows)
    fitness1 = CalFitness(obj1, con1)
    fitness2 = CalFitness(obj2, con2)

    for it in range(gen):
        print(f" Gen {it+1}/{gen}")
        # tournament_selection dùng rng
        pool1 = tournament_selection(2, pop_size, fitness1, rng)
        pool2 = tournament_selection(2, pop_size, fitness2, rng)
        # GA dùng rng
        Offspring1 = GA(pop1[pool1], n_route, lower, upper, rng)
        Offspring2 = GA(pop2[pool2], n_route, lower, upper, rng)
        # OBL dùng rng
        if rng.random() < p_obl:
            Offspring1 = np.vstack([Offspring1, obl(Offspring1, lower, upper, n_route, rng)])
            Offspring2 = np.vstack([Offspring2, obl(Offspring2, lower, upper, n_route, rng)])
        pop3 = np.vstack([pop1, Offspring1, Offspring2])
        pop4 = np.vstack([pop2, Offspring1, Offspring2])
        obj3, con3, *_ = cal_obj_con(pop3, route, n_route, pop3.shape[0], point, time_windows)
        obj4, con4, *_ = cal_obj_con(pop4, route, n_route, pop4.shape[0], point, time_windows)
        fitness3 = CalFitness(obj3, con3)
        fitness4 = CalFitness(obj4, con4)
        pop1, fitness1 = environmental_selection(pop3, fitness3, pop_size, obj3)
        pop2, fitness2 = environmental_selection(pop4, fitness4, pop_size, obj4)

    # tính lại obj/con khớp pop1
    obj1, con1, *_ = cal_obj_con(pop1, route, n_route, pop1.shape[0], point, time_windows)

    feasible = fitness1 < 1.0
    if feasible.any():
        best_pop = pop1[feasible]
        best_obj = obj1[feasible]
    else:
        k = int(np.argmin(fitness1))
        best_pop = pop1[[k]]
        best_obj = obj1[[k]]

    # TOPSIS kiểu MATLAB
    obj_sel, idx_topsis, _ = topsis_matlab(best_obj)
    x_best = best_pop[idx_topsis:idx_topsis+1]

    # gọi lại để lấy point_cha_sort (phục vụ vẽ MCV)
    (_fit_b, _con_b,
     _rst, _rse, _cst, _cse, _cet, _cee,
     _ret, _ree, point_cha_b, point_cha_sort_b) = cal_obj_con(
        x_best, route, n_route, 1, point, time_windows
    )

    print("Best objective (cost, TW violation):", obj_sel)

    # Vẽ
    plot_uav_route(point, route, start_point)
    plot_mcv_route(point, route, point_cha_sort_b, start_point)

if __name__ == "__main__":
    main()