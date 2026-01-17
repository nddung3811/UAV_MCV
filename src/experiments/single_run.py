from __future__ import annotations
import numpy as np
import os

from src import config
from src.io.mat import load_mat
from src.algorithms.selection import tournament_selection, environmental_selection
from src.algorithms.ga import genetic_operator
from src.algorithms.obl import opposition_based_learning
from src.decision.topsis import topsis_matlab
from src.visualization.routes import (
     plot_uav_route, plot_mcv_charging_detailed
)

from src.core.initialize import initialize_population
from src.core.fitness import evaluate_objectives_constraints, compute_fitness


# =========================================================
# Utilities
# =========================================================

def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _build_route(bestroute: np.ndarray,
                 bestbreak: np.ndarray,
                 depot_id: int) -> np.ndarray:
    br = np.atleast_1d(bestbreak).astype(int).ravel()
    K = int(np.atleast_1d(bestroute).size)
    br_full = np.concatenate(([0], br, [K]))

    route = [depot_id]
    for i in range(len(br_full) - 1):
        a = br_full[i] + 1
        b = br_full[i + 1]
        if a <= b:
            route.extend(bestroute[a - 1:b].tolist())
        route.append(depot_id)

    return np.asarray(route, dtype=int)


def _bounds_for_edges(route: np.ndarray,
                      points: np.ndarray,
                      depot: np.ndarray,
                      n_vehicle: int,
                      t_charge_min: float,
                      t_charge_max: float,
                      scope: float = 6.0):

    n_edges = len(route) - 1
    depot_id = points.shape[0] + 1

    lower = np.zeros(3 * n_edges)
    upper = np.zeros(3 * n_edges)

    lower[:n_edges] = 0
    upper[:n_edges] = n_vehicle

    for i in range(n_edges):
        u, v = int(route[i]), int(route[i + 1])
        if u == depot_id:
            d = _euclidean(points[v - 1], depot)
            lo, hi = 0.0, (d - scope) / d
        elif v == depot_id:
            d = _euclidean(points[u - 1], depot)
            lo, hi = scope / d, 1.0
        else:
            d = _euclidean(points[u - 1], points[v - 1])
            lo, hi = scope / d, (d - scope) / d

        lower[n_edges + i] = np.clip(lo, 0.0, 1.0)
        upper[n_edges + i] = np.clip(hi, 0.0, 1.0)

    lower[2 * n_edges:] = t_charge_min
    upper[2 * n_edges:] = t_charge_max

    return lower, upper


# =========================================================
# Main experiment
# =========================================================

def run_experiment(seed: int):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    depot = np.asarray(config.START_POINT, dtype=float)
    n_vehicle = int(config.N_VEHICLE)


    pop_size = 200
    generations = 200
    p_obl = 0.8
    t_charge_min, t_charge_max = 5.0, 30.0

    instance = getattr(config, "INSTANCE", "data/point3.mat")
    data = load_mat(instance)

    points = np.asarray(data["point"], dtype=float)
    bestroute = np.asarray(data["bestroute"], dtype=int).ravel()
    bestbreak = np.asarray(data["bestbreak"], dtype=int).ravel()
    time_windows = np.asarray(data["time_windows"]) if "time_windows" in data else None
    n_uav = len(bestbreak) + 1

    depot_id = points.shape[0] + 1
    route = _build_route(bestroute, bestbreak, depot_id)
    n_edges = len(route) - 1

    lower, upper = _bounds_for_edges(
        route, points, depot,
        n_vehicle, t_charge_min, t_charge_max
    )

    # Initialization
    pop1 = initialize_population(pop_size, n_edges, lower, upper, rng)
    pop2 = initialize_population(pop_size, n_edges, lower, upper, rng)

    obj1, con1, *_ = evaluate_objectives_constraints(
        pop1, route, n_edges, pop_size, points, time_windows
    )
    obj2, con2, *_ = evaluate_objectives_constraints(
        pop2, route, n_edges, pop_size, points, time_windows
    )

    fitness1 = compute_fitness(obj1, con1)
    fitness2 = compute_fitness(obj2, con2)

    # Evolution
    for it in range(generations):
        print(f"Generation {it + 1}/{generations}")

        pool1 = tournament_selection(2, pop_size, fitness1, rng)
        pool2 = tournament_selection(2, pop_size, fitness2, rng)

        off1 = genetic_operator(pop1[pool1], n_edges, lower, upper, rng)
        off2 = genetic_operator(pop2[pool2], n_edges, lower, upper, rng)

        if rng.random() < p_obl:
            off1 = np.vstack([
                off1,
                opposition_based_learning(off1, lower, upper, n_edges, rng)
            ])
            off2 = np.vstack([
                off2,
                opposition_based_learning(off2, lower, upper, n_edges, rng)
            ])

        pop_all_1 = np.vstack([pop1, off1, off2])
        pop_all_2 = np.vstack([pop2, off1, off2])

        obj_all_1, con_all_1, *_ = evaluate_objectives_constraints(
            pop_all_1, route, n_edges, pop_all_1.shape[0], points, time_windows
        )
        obj_all_2, con_all_2, *_ = evaluate_objectives_constraints(
            pop_all_2, route, n_edges, pop_all_2.shape[0], points, time_windows
        )

        pop1, fitness1 = environmental_selection(
            pop_all_1,
            compute_fitness(obj_all_1, con_all_1),
            pop_size,
            obj_all_1,
        )
        pop2, fitness2 = environmental_selection(
            pop_all_2,
            compute_fitness(obj_all_2, con_all_2),
            pop_size,
            obj_all_2,
        )

    # Final selection
    obj_final, con_final, *_ = evaluate_objectives_constraints(
        pop1, route, n_edges, pop1.shape[0], points, time_windows
    )

    feasible = fitness1 < 1.0
    if feasible.any():
        cand_pop = pop1[feasible]
        cand_obj = obj_final[feasible]
    else:
        k = int(np.argmin(fitness1))
        cand_pop = pop1[[k]]
        cand_obj = obj_final[[k]]

    obj_sel, idx, _ = topsis_matlab(cand_obj)

    log_line = (
        f"seed={seed}, "
        f"MCV={n_vehicle}, "
        f"best_objective={obj_sel.tolist()}"
    )
    print(log_line)

    os.makedirs("results", exist_ok=True)
    with open("results/best_objective.log", "a") as f:
        f.write(log_line + "\n")

    # Visualization
    x_best = cand_pop[idx:idx + 1]
    (_, _, _, _, _, _, _, _, _, _, _, point_cha_sort, assign_info) = (
        evaluate_objectives_constraints(
            x_best, route, n_edges, 1, points, time_windows
        )
    )

    tag = f"seed{seed}_mcv{n_vehicle}"

    plot_uav_route(points, route, depot,
                   save_path=f"results/{tag}_uav_route.png")


    plot_mcv_charging_detailed(
        points,
        route,
        depot,
        assign_info[0],
        title=f"MCV charging (seed={seed},  MCV={n_vehicle})",
        save_path=f"results/{tag}_mcv_detail.png"
    )


if __name__ == "__main__":
    run_experiment(seed=1)
