from __future__ import annotations
import numpy as np
import os

from .. import config
from ..io.mat import load_mat
from ..moea.selection import tournament_selection, environmental_selection
from ..moea.ga import GA
from ..moea.obl import obl
from ..metrics.topsis import topsis_matlab
from ..viz.routes import (
    plot_mcv_route, plot_uav_route,
    plot_uav_routes_detailed, plot_mcv_charging_detailed
)

from .initialize import initialize
from .fitness import cal_obj_con, CalFitness


# =============== Utilities ===============
def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def _build_route(bestroute: np.ndarray, bestbreak: np.ndarray, nn: int) -> np.ndarray:
    br = np.atleast_1d(bestbreak).astype(int).ravel()
    K = int(np.atleast_1d(bestroute).size)
    br_full = np.concatenate(([0], br, [K]))

    rt = [nn]
    for i in range(len(br_full) - 1):
        a = br_full[i] + 1
        b = br_full[i + 1]
        if a <= b:
            seg = np.atleast_1d(bestroute)[a - 1:b]
            rt.extend(seg.tolist())
        rt.append(nn)
    return np.array(rt, dtype=int)


def _bounds_for_edges(route, point, start_point,
                      n_vehicle, t_char_min, t_char_max, scope=6.0):
    n_route = len(route) - 1
    nn = point.shape[0] + 1

    lower = np.zeros(3 * n_route)
    upper = np.zeros(3 * n_route)

    lower[:n_route] = 0
    upper[:n_route] = n_vehicle

    for i in range(n_route):
        u, v = int(route[i]), int(route[i + 1])
        if u == nn:
            d = _euclid(point[v - 1], start_point)
            lo, hi = 0.0, (d - scope) / d
        elif v == nn:
            d = _euclid(point[u - 1], start_point)
            lo, hi = scope / d, 1.0
        else:
            d = _euclid(point[u - 1], point[v - 1])
            lo, hi = scope / d, (d - scope) / d

        lower[n_route + i] = np.clip(lo, 0, 1)
        upper[n_route + i] = np.clip(hi, 0, 1)

    lower[2 * n_route:] = t_char_min
    upper[2 * n_route:] = t_char_max
    return lower, upper


# =============== MAIN: chạy 1 experiment ===============
def main(seed: int):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    start_point = np.array(config.START_POINT, dtype=float)
    n_vehicle = config.N_VEHICLE

    t_char_min, t_char_max = 5.0, 30.0
    p_obl = 0.8
    pop_size = 200
    gen = 200

    instance = getattr(config, "INSTANCE", "data/point3.mat")
    data = load_mat(instance)

    point = np.array(data["point"], dtype=float)
    bestroute = np.array(data["bestroute"], dtype=int).ravel()
    bestbreak = np.array(data["bestbreak"], dtype=int).ravel()
    time_windows = np.array(data["time_windows"]) if "time_windows" in data else None

    nn = point.shape[0] + 1
    route = _build_route(bestroute, bestbreak, nn)
    n_route = len(route) - 1

    lower, upper = _bounds_for_edges(
        route, point, start_point,
        n_vehicle, t_char_min, t_char_max
    )

    pop1 = initialize(pop_size, n_route, lower, upper, rng=rng)
    pop2 = initialize(pop_size, n_route, lower, upper, rng=rng)

    obj1, con1, *_ = cal_obj_con(pop1, route, n_route, pop_size, point, time_windows)
    obj2, con2, *_ = cal_obj_con(pop2, route, n_route, pop_size, point, time_windows)
    fitness1 = CalFitness(obj1, con1)
    fitness2 = CalFitness(obj2, con2)

    for it in range(gen):
        print(f" Gen {it+1}/{gen}")

        pool1 = tournament_selection(2, pop_size, fitness1, rng)
        pool2 = tournament_selection(2, pop_size, fitness2, rng)

        off1 = GA(pop1[pool1], n_route, lower, upper, rng)
        off2 = GA(pop2[pool2], n_route, lower, upper, rng)

        if rng.random() < p_obl:
            off1 = np.vstack([off1, obl(off1, lower, upper, n_route, rng)])
            off2 = np.vstack([off2, obl(off2, lower, upper, n_route, rng)])

        pop3 = np.vstack([pop1, off1, off2])
        pop4 = np.vstack([pop2, off1, off2])

        obj3, con3, *_ = cal_obj_con(pop3, route, n_route, pop3.shape[0], point, time_windows)
        obj4, con4, *_ = cal_obj_con(pop4, route, n_route, pop4.shape[0], point, time_windows)

        pop1, fitness1 = environmental_selection(pop3, CalFitness(obj3, con3), pop_size, obj3)
        pop2, fitness2 = environmental_selection(pop4, CalFitness(obj4, con4), pop_size, obj4)

    obj1, con1, *_ = cal_obj_con(pop1, route, n_route, pop1.shape[0], point, time_windows)
    feasible = fitness1 < 1.0

    if feasible.any():
        best_pop = pop1[feasible]
        best_obj = obj1[feasible]
    else:
        k = int(np.argmin(fitness1))
        best_pop = pop1[[k]]
        best_obj = obj1[[k]]

    obj_sel, idx, _ = topsis_matlab(best_obj)

    # ===== LOG ĐÚNG DÒNG BẠN CẦN =====
    log_line = (
        f"seed={seed}, "
        f"UAV={config.N_UAV}, "
        f"MCV={config.N_VEHICLE}, "
        f"Best objective (cost, TW violation): "
        f"{obj_sel.tolist()}"
    )

    print(log_line)

    os.makedirs("results", exist_ok=True)
    with open("results/best_objective.log", "a") as f:
        f.write(log_line + "\n")

    # ===== VẼ =====
    x_best = best_pop[idx:idx+1]
    (_, _, _, _, _, _, _, _, _, _, _, point_cha_sort, assign_info) = cal_obj_con(
        x_best, route, n_route, 1, point, time_windows
    )

    tag = f"seed{seed}_uav{config.N_UAV}_mcv{config.N_VEHICLE}"

    plot_uav_route(point, route, start_point,
                   save_path=f"results/{tag}_uav_route.png")

    plot_mcv_route(point, route, point_cha_sort, start_point,
                   save_path=f"results/{tag}_mcv_route.png")

    plot_uav_routes_detailed(point, route, start_point, assign_info[0],
                             save_path=f"results/{tag}_uav_detail.png")

    plot_mcv_charging_detailed(
        point,
        route,
        start_point,
        assign_info[0],
        title=f"MCV charging (seed={seed}, UAV={config.N_UAV}, MCV={config.N_VEHICLE})",
        save_path=f"results/{tag}_mcv_detail.png"
    )


# =============== RUN ALL ===============
if __name__ == "__main__":

    SEEDS = [0, 1, 2, 3, 4, 5 , 6, 7, 8, 9]

    for seed in SEEDS:
        for uav in range(1, 4):      # UAV: 1 → 3
            for mcv in range(3, 8):  # MCV: 3 → 7

                config.N_UAV = uav
                config.N_VEHICLE = mcv

                print(f"\n===== RUN seed={seed}, UAV={uav}, MCV={mcv} =====")

                main(seed)
