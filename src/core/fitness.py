from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from .. import config


# =========================================================
# Helpers
# =========================================================

def split_edges_by_uav(n_edges: int, n_uav: int) -> list[tuple[int, int]]:

    n_uav = int(n_uav)
    base = n_edges // n_uav
    rem = n_edges % n_uav

    blocks = []
    start = 0
    for i in range(n_uav):
        size = base + (1 if i < rem else 0)
        blocks.append((start, start + size))
        start += size

    return blocks


# =========================================================
# Objective & constraint evaluation
# =========================================================

def evaluate_objectives_constraints(
    population: np.ndarray,
    route: np.ndarray,          # 1-based, depot = n_point + 1
    n_edges: int,
    pop_size: int,
    points: np.ndarray,         # (n_point, 2), without depot
    time_windows: np.ndarray | None,
):
    """
    Objective & constraint evaluation for UAV–MCV cooperation.
    Number of UAVs is FIXED and defined in config.
    """

    n_points = points.shape[0]
    n_vehicle = int(config.N_VEHICLE)
    n_uav = max(1, int(getattr(config, "N_UAV", 1)))
    depot = np.asarray(config.START_POINT, dtype=float)

    # ---------------- Physical parameters ----------------
    uav_capacity = 200.0
    uav_speed = 2.2
    uav_energy_rate = 1.0
    uav_task_energy = 2.0

    hang_energy = 0.5
    vehicle_energy = 1.0
    vehicle_speed = 1.5

    task_time = 5.0
    charge_rate = 5.0
    electricity_price = 5.0
    efficiency = 0.3
    vehicle_cost = 5000.0

    time_delay = 0.5
    energy_delay = 0.5

    # ---------------- Buffers ----------------
    route_start_time = np.zeros((pop_size, n_edges))
    route_end_time   = np.zeros((pop_size, n_edges))
    route_start_energy = np.zeros((pop_size, n_edges))
    route_end_energy   = np.zeros((pop_size, n_edges))

    charge_start_time = np.zeros((pop_size, n_edges))
    charge_end_time   = np.zeros((pop_size, n_edges))
    charge_start_energy = np.zeros((pop_size, n_edges))
    charge_end_energy   = np.zeros((pop_size, n_edges))

    objectives = np.zeros((pop_size, 2))
    constraints = np.zeros((pop_size, 3))

    charge_points = [None] * pop_size
    charge_points_sorted = [None] * pop_size
    assignment_info = [None] * pop_size

    # Extend points with depot
    points_ext = np.vstack([points, depot[None, :]])
    depot_idx = n_points + 1

    # FIXED UAV segmentation
    uav_blocks = split_edges_by_uav(n_edges, n_uav)

    uav_id_per_edge = np.zeros(n_edges, dtype=int)
    for uid, (L, R) in enumerate(uav_blocks, start=1):
        uav_id_per_edge[L:R] = uid

    # =====================================================
    # Main loop
    # =====================================================

    for i in range(pop_size):
        constraint_i = np.zeros(3)
        pcs_list = []

        # ---------- UAV traversal ----------
        for j in range(n_edges):
            u = int(route[j])
            v = int(route[j + 1])

            is_uav_start = any(j == L for (L, _) in uav_blocks)

            if is_uav_start or u == depot_idx:
                route_start_time[i, j] = 0.0
                route_start_energy[i, j] = uav_capacity
            else:
                route_start_time[i, j] = route_end_time[i, j - 1] + task_time
                route_start_energy[i, j] = (
                    route_end_energy[i, j - 1] - task_time * uav_task_energy
                )
                if route_start_energy[i, j] < 0.1 * uav_capacity:
                    constraint_i[0] += 0.1 * uav_capacity - route_start_energy[i, j]

            p1 = points_ext[u - 1] if u != depot_idx else depot
            p2 = points_ext[v - 1] if v != depot_idx else depot
            edge_length = float(np.linalg.norm(p1 - p2))

            # ---------- No charging ----------
            if int(population[i, j]) == 0:
                pcs_list.append([0.0, 0.0])

                route_end_time[i, j] = route_start_time[i, j] + edge_length / uav_speed
                route_end_energy[i, j] = (
                    route_start_energy[i, j]
                    - edge_length / uav_speed * uav_energy_rate
                )

                if route_end_energy[i, j] < 0.1 * uav_capacity:
                    constraint_i[0] += 0.1 * uav_capacity - route_end_energy[i, j]

            # ---------- With charging ----------
            else:
                ratio = float(population[i, n_edges + j])
                charge_time = float(population[i, 2 * n_edges + j])

                p_charge = ratio * (p2 - p1) + p1
                pcs_list.append(p_charge.tolist())

                dist_to_charge = float(np.linalg.norm(p1 - p_charge))

                charge_start_time[i, j] = (
                    route_start_time[i, j]
                    + dist_to_charge / uav_speed
                    + time_delay
                )
                charge_end_time[i, j] = charge_start_time[i, j] + charge_time

                route_end_time[i, j] = (
                    route_start_time[i, j]
                    + edge_length / uav_speed
                    + charge_time
                    + time_delay
                )

                charge_start_energy[i, j] = (
                    route_start_energy[i, j]
                    - dist_to_charge / uav_speed * uav_energy_rate
                    - energy_delay
                )

                charge_end_energy[i, j] = (
                    charge_start_energy[i, j]
                    + charge_time * charge_rate
                    - charge_time * hang_energy
                )

                route_end_energy[i, j] = (
                    route_start_energy[i, j]
                    - edge_length / uav_speed * uav_energy_rate
                    + charge_time * (charge_rate - hang_energy)
                    - energy_delay
                )

                if route_end_energy[i, j] < 0.1 * uav_capacity:
                    constraint_i[0] += 0.1 * uav_capacity - route_end_energy[i, j]
                if charge_start_energy[i, j] < 0.1 * uav_capacity:
                    constraint_i[0] += 0.1 * uav_capacity - charge_start_energy[i, j]
                if charge_end_energy[i, j] > uav_capacity:
                    constraint_i[1] += charge_end_energy[i, j] - uav_capacity

        pcs_arr = np.asarray(pcs_list, dtype=float)
        charge_points[i] = pcs_arr

        # ---------- MCV routing ----------
        vehicle_constraints = np.zeros(n_vehicle)
        vehicle_times = []
        sorted_pts = [depot.copy()]

        for k in range(1, n_vehicle + 1):
            idx = np.where(population[i, :n_edges].astype(int) == k)[0]
            if idx.size == 0:
                vehicle_times.append(0.0)
                continue

            pts = pcs_arr[idx]
            t_start = charge_start_time[i, idx]
            order = np.argsort(t_start)

            pts = pts[order]
            t_start = t_start[order]
            t_end = charge_end_time[i, idx][order]

            dist = (
                np.linalg.norm(depot - pts[0])
                + np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1))
                + np.linalg.norm(depot - pts[-1])
            )
            vehicle_times.append(dist / vehicle_speed)

            if pts.shape[0] >= 2:
                travel = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
                gap = t_start[1:] - t_end[:-1]
                v_req = travel / gap
                excess = v_req - vehicle_speed
                vehicle_constraints[k - 1] = (
                    np.sum(excess[excess > 0.0]) + np.sum(-v_req[v_req < 0.0])
                )

            sorted_pts.extend(pts.tolist())

        sorted_pts.append(depot.copy())
        charge_points_sorted[i] = np.asarray(sorted_pts)

        # ---------- Objectives ----------
        used_vehicles = np.unique(population[i, :n_edges].astype(int))
        used_vehicles = used_vehicles[used_vehicles != 0]

        num_charges = int(np.sum(population[i, :n_edges] != 0))

        objectives[i, 0] = (
            np.sum(vehicle_times) * vehicle_energy
            + np.sum(population[i, 2 * n_edges:3 * n_edges]) * charge_rate / efficiency
            + energy_delay * num_charges
        ) * electricity_price + vehicle_cost * len(used_vehicles)

        objectives[i, 1] = 0.0
        constraints[i, 2] = np.sum(vehicle_constraints)

        # ---------- Time windows ----------
        if time_windows is not None:
            for e in range(1, n_edges + 1):
                if int(route[e]) != depot_idx:
                    late = route_end_time[i, e - 1] - time_windows[int(route[e]) - 1]
                    if late > 0:
                        objectives[i, 1] += late

        constraints[i, :] = constraint_i
        uav_routes_nodes = {}
        for uid, (L, R) in enumerate(uav_blocks, start=1):
            # edges [L, R-1] correspond to nodes route[L] -> route[R]
            nodes = route[L:R + 1].tolist()
            uav_routes_nodes[uid] = nodes
        mcv_charge_points_sorted = {}

        for k in range(1, n_vehicle + 1):
            idx = np.where(population[i, :n_edges].astype(int) == k)[0]
            if idx.size == 0:
                continue

            pts = pcs_arr[idx]
            t_start = charge_start_time[i, idx]

            order = np.argsort(t_start)
            pts = pts[order]

            mcv_charge_points_sorted[k] = pts.tolist()

        # ---------- Plot info ----------
        assignment_info[i] = {
            "uav_blocks": uav_blocks,
            "uav_id_per_edge": uav_id_per_edge.tolist(),
            "uav_routes_nodes": uav_routes_nodes,
            "mcv_id_per_edge": population[i, :n_edges].astype(int).tolist(),
            "mcv_charge_points_sorted": mcv_charge_points_sorted,
        }

    return (
        objectives, constraints,
        route_start_time, route_start_energy,
        charge_start_time, charge_start_energy,
        charge_end_time, charge_end_energy,
        route_end_time, route_end_energy,
        charge_points, charge_points_sorted,
        assignment_info,
    )


# =========================================================
# Fitness assignment (SPEA2-style)
# =========================================================

def compute_fitness(
    objectives: np.ndarray,
    constraints: np.ndarray | None = None,
) -> np.ndarray:
    """
    Constraint-dominated fitness with density estimation.
    """
    n = objectives.shape[0]
    cv = np.zeros(n) if constraints is None else np.maximum(constraints, 0.0).sum(axis=1)

    dominates = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if cv[i] < cv[j]:
                dominates[i, j] = True
            elif cv[i] > cv[j]:
                dominates[j, i] = True
            else:
                better = (objectives[i] < objectives[j]).any()
                worse = (objectives[i] > objectives[j]).any()
                if better and not worse:
                    dominates[i, j] = True
                elif worse and not better:
                    dominates[j, i] = True

    strength = dominates.sum(axis=1)
    raw_fitness = np.zeros(n)
    for i in range(n):
        raw_fitness[i] = strength[dominates[:, i]].sum()

    dist = cdist(objectives, objectives)
    np.fill_diagonal(dist, np.inf)
    dist = np.sort(dist, axis=1)
    density = 1.0 / (dist[:, int(np.sqrt(n))] + 2.0)

    return raw_fitness + density
