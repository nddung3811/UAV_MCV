    # src/uav_mcv/fitness.py
    from __future__ import annotations
    import numpy as np
    from .. import config


    def split_edges_by_uav(n_route: int, n_uav: int) -> list[tuple[int, int]]:
        """
        Chia các edge [0..n_route-1] thành n_uav block liên tiếp.
        Return list[(L,R)] theo kiểu Python slice: L inclusive, R exclusive.
        VD: n_route=20, n_uav=2 -> [(0,10),(10,20)]
        """
        n_uav = max(1, int(n_uav))
        base = n_route // n_uav
        rem = n_route % n_uav
        blocks = []
        s = 0
        for u in range(n_uav):
            k = base + (1 if u < rem else 0)
            blocks.append((s, s + k))
            s += k
        return blocks


    def cal_obj_con(pop: np.ndarray,
                    route: np.ndarray,         # 1-based, có depot = n_point+1
                    n_route: int,
                    pop_size: int,
                    point: np.ndarray,         # shape (n_point,2) — CHƯA gồm depot
                    time_windows: np.ndarray | None):

        n_point   = point.shape[0]
        n_vehicle = config.N_VEHICLE
        n_uav     = max(1, int(getattr(config, "N_UAV", 1)))   # <<< dùng để vẽ / chia block
        start_pt  = np.array(config.START_POINT, dtype=float)

        # ====== Tham số vật lý (giống hệt file .m) ======
        ca_uav = 200.0
        v_uav = 2.2
        v_ele_uav = 1.0
        v_ele_uav_task = 2.0
        v_ele_hang = 0.5
        v_ele_vehicle = 1.0
        v_vehicles = 1.5
        time_task = 5.0
        v_cha = 5.0
        ele_price = 5.0
        eff = 0.3
        cost_ve = 5000.0
        t_d = 0.5
        q_d = 0.5

        # ====== Chuẩn bị mảng kết quả ======
        route_start_time = np.zeros((pop_size, n_route))
        route_start_ele  = np.zeros((pop_size, n_route))
        cha_start_time   = np.zeros((pop_size, n_route))
        cha_end_time     = np.zeros((pop_size, n_route))
        route_end_time   = np.zeros((pop_size, n_route))

        cha_start_ele    = np.zeros((pop_size, n_route))
        cha_end_ele      = np.zeros((pop_size, n_route))
        route_end_ele    = np.zeros((pop_size, n_route))

        fit = np.zeros((pop_size, 2))
        con = np.zeros((pop_size, 3))

        point_cha      = [None] * pop_size
        point_cha_sort = [None] * pop_size

        # NEW: info để vẽ (mỗi cá thể 1 dict)
        assign_info    = [None] * pop_size

        # mở rộng point để truy cập depot 1-based = n_point+1
        Pext = np.vstack([point, start_pt[None, :]])
        nn = n_point + 1

        # UAV blocks cố định theo n_route
        uav_blocks = split_edges_by_uav(n_route, n_uav)

        # map edge j -> uav_id (1..n_uav)
        uav_id_per_edge = np.zeros(n_route, dtype=int)
        for uid, (L, R) in enumerate(uav_blocks, start=1):
            if R > L:
                uav_id_per_edge[L:R] = uid

        for i in range(pop_size):
            con_i = np.zeros(3)
            pcs_list = []   # (n_route,2), [0,0] nếu không sạc

            for j in range(n_route):
                u = int(route[j])
                v = int(route[j + 1])

                is_new_uav = any(j == L for (L, R) in uav_blocks)

                if is_new_uav:
                    route_start_time[i, j] = 0.0
                    route_start_ele[i, j] = ca_uav
                if u == n_point + 1:
                    route_start_time[i, j] = 0.0
                    route_start_ele[i, j]  = ca_uav
                else:
                    route_start_time[i, j] = route_end_time[i, j - 1] + time_task
                    route_start_ele[i, j]  = route_end_ele[i, j - 1] - time_task * v_ele_uav_task
                    if route_start_ele[i, j] < ca_uav * 0.1:
                        con_i[0] += (ca_uav * 0.1 - route_start_ele[i, j])

                p1 = Pext[u - 1] if u != nn else start_pt
                p2 = Pext[v - 1] if v != nn else start_pt
                d_start_end = float(np.linalg.norm(p1 - p2))

                if int(pop[i, j]) == 0:
                    pcs_list.append([0.0, 0.0])
                    cha_start_time[i, j] = 0.0
                    cha_end_time[i, j]   = 0.0
                    route_end_time[i, j] = route_start_time[i, j] + d_start_end / v_uav

                    cha_start_ele[i, j] = 0.0
                    cha_end_ele[i, j]   = 0.0
                    route_end_ele[i, j] = route_start_ele[i, j] - d_start_end / v_uav * v_ele_uav
                    if route_end_ele[i, j] < ca_uav * 0.1:
                        con_i[0] += (ca_uav * 0.1 - route_end_ele[i, j])

                else:
                    ratio = float(pop[i, n_route + j])
                    tcha  = float(pop[i, 2 * n_route + j])

                    pcha = ratio * (p2 - p1) + p1
                    pcs_list.append(pcha.tolist())

                    d_start_cha = float(np.linalg.norm(p1 - pcha))
                    cha_start_time[i, j] = route_start_time[i, j] + d_start_cha / v_uav + t_d
                    cha_end_time[i, j]   = cha_start_time[i, j] + tcha
                    route_end_time[i, j] = route_start_time[i, j] + d_start_end / v_uav + tcha + t_d

                    cha_start_ele[i, j] = route_start_ele[i, j] - d_start_cha / v_uav * v_ele_uav - q_d
                    cha_end_ele[i, j]   = cha_start_ele[i, j] + tcha * v_cha - tcha * v_ele_hang
                    route_end_ele[i, j] = route_start_ele[i, j] - d_start_end / v_uav * v_ele_uav + tcha * (v_cha - v_ele_hang) - q_d

                    if route_end_ele[i, j] < ca_uav * 0.1:
                        con_i[0] += (ca_uav * 0.1 - route_end_ele[i, j])
                    if cha_start_ele[i, j] < ca_uav * 0.1:
                        con_i[0] += (ca_uav * 0.1 - cha_start_ele[i, j])
                    if cha_end_ele[i, j] > ca_uav:
                        con_i[1] += (cha_end_ele[i, j] - ca_uav)

            pcs_arr = np.asarray(pcs_list, dtype=float)
            point_cha[i] = pcs_arr

            # ----- gom theo từng MCV và sắp xếp theo thời gian sạc -----
            con_per_vehicle = np.zeros(n_vehicle)
            pcs_sorted_for_plot = [start_pt.copy()]

            for k in range(1, n_vehicle + 1):
                idx = np.where(pop[i, :n_route].astype(int) == k)[0]
                if idx.size == 0:
                    t_vehicle_k = 0.0
                else:
                    pts_k = pcs_arr[idx, :]
                    tstart_k = cha_start_time[i, idx]
                    order = np.argsort(tstart_k)
                    pts_k = pts_k[order, :]

                    seg = np.linalg.norm(pts_k[1:] - pts_k[:-1], axis=1) if pts_k.shape[0] >= 2 else np.array([])
                    d_k = np.linalg.norm(start_pt - pts_k[0]) + seg.sum() + np.linalg.norm(start_pt - pts_k[-1])
                    t_vehicle_k = d_k / v_vehicles

                    tend_k = cha_end_time[i, idx][order]
                    if pts_k.shape[0] >= 2:
                        d_mid = np.linalg.norm(pts_k[1:] - pts_k[:-1], axis=1)
                        t_gap = tstart_k[order][1:] - tend_k[:-1]
                        v_ideal = d_mid / t_gap
                        xu = v_ideal - v_vehicles
                        con_per_vehicle[k - 1] = np.sum(xu[xu > 0.0]) + np.sum(-v_ideal[v_ideal < 0.0])

                    pcs_sorted_for_plot.extend(pts_k.tolist())

                if k == 1:
                    t_vehicle = [t_vehicle_k]
                else:
                    t_vehicle.append(t_vehicle_k)

            pcs_sorted_for_plot.append(start_pt.copy())
            point_cha_sort[i] = np.asarray(pcs_sorted_for_plot, dtype=float)

            # ----- mục tiêu & ràng buộc tổng hợp -----
            num_ids = np.unique(pop[i, :n_route].astype(int))
            num_ids = num_ids[num_ids != 0]
            num_charges = int(np.sum(pop[i, :n_route] != 0))

            fit[i, 0] = (np.sum(t_vehicle) * v_ele_vehicle + np.sum(pop[i, 2 * n_route:3 * n_route]) * v_cha / eff + q_d * num_charges) * ele_price \
                        + cost_ve * (len(num_ids))
            fit[i, 1] = 0.0

            con_i[2] = np.sum(con_per_vehicle)

            if time_windows is not None:
                for z in range(1, n_route + 1):
                    if int(route[z]) != nn:
                        late = route_end_time[i, z - 1] - float(time_windows[int(route[z]) - 1])
                        if late > 0:
                            fit[i, 1] += late

            con[i, :] = con_i

            # ===================== NEW: build info for plotting =====================
            # UAV: node list per UAV (the nodes the UAV "flies through" in its block)
            uav_routes_nodes: dict[int, list[int]] = {}
            for uid, (L, R) in enumerate(uav_blocks, start=1):
                nodes = route[L:R+1].astype(int).tolist() if R > L else []
                uav_routes_nodes[uid] = nodes

            # MCV: which edges are assigned to each vehicle, and charging points along those edges
            mcv_id_per_edge = pop[i, :n_route].astype(int)  # 0..n_vehicle
            mcv_charge_edges: dict[int, list[int]] = {}
            mcv_charge_points: dict[int, list[list[float]]] = {}
            mcv_charge_edges_sorted: dict[int, list[int]] = {}

            for k in range(1, n_vehicle + 1):
                idx = np.where(mcv_id_per_edge == k)[0]
                if idx.size == 0:
                    mcv_charge_edges[k] = []
                    mcv_charge_points[k] = []
                    mcv_charge_edges_sorted[k] = []
                    continue

                # chỉ lấy edge thực sự có sạc (pcs != [0,0])
                pts = pcs_arr[idx, :]
                mask_charge = ~((pts[:, 0] == 0.0) & (pts[:, 1] == 0.0))
                idx2 = idx[mask_charge]
                pts2 = pts[mask_charge]

                if idx2.size == 0:
                    mcv_charge_edges[k] = []
                    mcv_charge_points[k] = []
                    mcv_charge_edges_sorted[k] = []
                    continue

                mcv_charge_edges[k] = idx2.astype(int).tolist()
                mcv_charge_points[k] = pts2.astype(float).tolist()

                # sort theo thời gian bắt đầu sạc để vẽ đường MCV theo thời
                tstart2 = cha_start_time[i, idx2]
                order2 = np.argsort(tstart2)
                mcv_charge_edges_sorted[k] = idx2[order2].astype(int).tolist()
                # đồng bộ points theo order2
                mcv_charge_points[k] = pts2[order2].astype(float).tolist()

            assign_info[i] = {
                "uav_blocks": uav_blocks,                      # [(L,R),...]
                "uav_id_per_edge": uav_id_per_edge.tolist(),   # len n_route
                "uav_routes_nodes": uav_routes_nodes,          # uid -> list[node 1-based]
                "mcv_id_per_edge": mcv_id_per_edge.tolist(),   # len n_route
                "mcv_charge_edges_sorted": mcv_charge_edges_sorted,  # k -> list[edge_j]
                "mcv_charge_points_sorted": mcv_charge_points,       # k -> list[[x,y]]
            }

        return (fit, con,
                route_start_time, route_start_ele,
                cha_start_time, cha_start_ele,
                cha_end_time, cha_end_ele,
                route_end_time, route_end_ele,
                point_cha, point_cha_sort,
                assign_info)


    def CalFitness(obj: np.ndarray, con: np.ndarray | None = None) -> np.ndarray:
        N = obj.shape[0]
        CV = np.zeros(N) if con is None else np.maximum(con, 0.0).sum(axis=1)

        Dominate = np.zeros((N, N), dtype=bool)
        for i in range(N - 1):
            for j in range(i + 1, N):
                if CV[i] < CV[j]:
                    Dominate[i, j] = True
                elif CV[i] > CV[j]:
                    Dominate[j, i] = True
                else:
                    k = int((obj[i] < obj[j]).any()) - int((obj[i] > obj[j]).any())
                    if k == 1:
                        Dominate[i, j] = True
                    elif k == -1:
                        Dominate[j, i] = True

        S = Dominate.sum(axis=1)
        R = np.zeros(N)
        for i in range(N):
            R[i] = S[Dominate[:, i]].sum()

        from scipy.spatial.distance import cdist
        D = cdist(obj, obj)
        np.fill_diagonal(D, np.inf)
        D = np.sort(D, axis=1)
        D = 1.0 / (D[:, int(np.sqrt(N))] + 2.0)

        return R + D
