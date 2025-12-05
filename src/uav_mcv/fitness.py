# src/uav_mcv/fitness.py
from __future__ import annotations
import numpy as np
from .. import config


def cal_obj_con(pop: np.ndarray,
                route: np.ndarray,         # 1-based, có depot = n_point+1
                n_route: int,
                pop_size: int,
                point: np.ndarray,         # shape (n_point,2) — CHƯA gồm depot
                time_windows: np.ndarray | None):

    n_point   = point.shape[0]
    n_vehicle = config.N_VEHICLE
    start_pt  = np.array(config.START_POINT, dtype=float)

    # ====== Tham số vật lý (giống hệt file .m) ======
    ca_uav = 200.0           # dung lượng pin UAV
    v_uav = 2.2              # vận tốc UAV (km/min, theo hệ toạ độ của tác giả)
    v_ele_uav = 1.0          # tiêu hao khi bay
    v_ele_uav_task = 2.0     # tiêu hao khi thực hiện nhiệm vụ (hover)
    v_ele_hang = 0.5         # tiêu hao khi sạc đang lơ lửng
    v_ele_vehicle = 1.0      # hệ số chuyển thời gian di chuyển MCV -> chi phí điện
    v_vehicles = 1.5         # vận tốc MCV (km/min)
    time_task = 5.0          # thời gian làm nhiệm vụ tại mỗi điểm
    v_cha = 5.0              # tốc độ nạp điện (/min)
    ele_price = 5.0          # giá điện (hệ số)
    eff = 0.3                # hiệu suất sạc
    cost_ve = 5000.0         # chi phí cố định mỗi MCV sử dụng
    t_d = 0.5                # overhead bắt/nhả sạc
    q_d = 0.5                # hao hụt khi cắm/rút

    # ====== Chuẩn bị mảng kết quả ======
    # (tất cả giữ kích thước [pop_size, n_route])
    route_start_time = np.zeros((pop_size, n_route))
    route_start_ele  = np.zeros((pop_size, n_route))
    cha_start_time   = np.zeros((pop_size, n_route))
    cha_end_time     = np.zeros((pop_size, n_route))
    route_end_time   = np.zeros((pop_size, n_route))

    cha_start_ele    = np.zeros((pop_size, n_route))
    cha_end_ele      = np.zeros((pop_size, n_route))
    route_end_ele    = np.zeros((pop_size, n_route))

    fit = np.zeros((pop_size, 2))     # 2 objectives: cost, TW_violation
    con = np.zeros((pop_size, 3))     # 3 constraints (tổng hợp như MATLAB)

    # danh sách toạ độ điểm sạc theo từng cá thể
    point_cha      = [None] * pop_size
    point_cha_sort = [None] * pop_size

    # mở rộng point để truy cập depot 1-based = n_point+1
    Pext = np.vstack([point, start_pt[None, :]])
    nn = n_point + 1

    for i in range(pop_size):
        # ----- tính state dọc tuyến -----
        con_i = np.zeros(3)
        pcs_list = []   # point_cha1: (n_route,2), 0 nếu không sạc
        for j in range(n_route):
            u = int(route[j])     # 1-based
            v = int(route[j + 1])

            if u == n_point + 1:
                # xuất phát từ depot
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
                # không sạc
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
                # có sạc trên cung j
                ratio = float(pop[i, n_route + j])          # tỉ lệ vị trí sạc trên đoạn
                tcha  = float(pop[i, 2 * n_route + j])      # thời gian sạc

                pcha = ratio * (p2 - p1) + p1               # toạ độ sạc
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
        # chuỗi vẽ: start -> (các điểm sạc theo thứ tự) -> start
        pcs_sorted_for_plot = [start_pt.copy()]

        for k in range(1, n_vehicle + 1):
            idx = np.where(pop[i, :n_route].astype(int) == k)[0]
            if idx.size == 0:
                t_vehicle_k = 0.0
            else:
                pts_k = pcs_arr[idx, :]                        # các điểm sạc của xe k
                tstart_k = cha_start_time[i, idx]              # thời điểm bắt đầu sạc
                order = np.argsort(tstart_k)                   # sắp xếp theo thời
                pts_k = pts_k[order, :]
                # tổng quãng đường: base -> p1 -> ... -> p_m -> base
                seg = np.linalg.norm(pts_k[1:] - pts_k[:-1], axis=1) if pts_k.shape[0] >= 2 else np.array([])
                d_k = np.linalg.norm(start_pt - pts_k[0]) + seg.sum() + np.linalg.norm(start_pt - pts_k[-1])
                t_vehicle_k = d_k / v_vehicles

                # ràng buộc tốc độ lý tưởng giữa 2 lần sạc liên tiếp
                tend_k = cha_end_time[i, idx][order]
                if pts_k.shape[0] >= 2:
                    d_mid = np.linalg.norm(pts_k[1:] - pts_k[:-1], axis=1)  # d1 trong .m
                    t_gap = tstart_k[order][1:] - tend_k[:-1]               # t3 trong .m
                    v_ideal = d_mid / t_gap
                    # vi phạm: > v_vehicles (dương) hoặc t_gap <= 0 (âm tốc độ)
                    xu = v_ideal - v_vehicles
                    con_per_vehicle[k - 1] = np.sum(xu[xu > 0.0]) + np.sum(-v_ideal[v_ideal < 0.0])

                # thêm vào chuỗi vẽ
                pcs_sorted_for_plot.extend(pts_k.tolist())

            # lưu thời gian MCV k
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

        # cửa sổ thời gian: chỉ cộng trễ khi đích tiếp theo không phải depot
        if time_windows is not None:
            for z in range(1, n_route + 1):
                if int(route[z]) != nn:
                    late = route_end_time[i, z - 1] - float(time_windows[int(route[z]) - 1])
                    if late > 0:
                        fit[i, 1] += late

        con[i, :] = con_i

    return (fit, con,
            route_start_time, route_start_ele,
            cha_start_time, cha_start_ele,
            cha_end_time, cha_end_ele,
            route_end_time, route_end_ele,
            point_cha, point_cha_sort)


def CalFitness(obj: np.ndarray, con: np.ndarray | None = None) -> np.ndarray:

    N = obj.shape[0]
    CV = np.zeros(N) if con is None else np.maximum(con, 0.0).sum(axis=1)

    # Dominate matrix
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

    # crowding distance kiểu sqrt(N)
    from scipy.spatial.distance import cdist
    D = cdist(obj, obj)
    np.fill_diagonal(D, np.inf)
    D = np.sort(D, axis=1)
    D = 1.0 / (D[:, int(np.sqrt(N))] + 2.0)

    f = R + D
    return f
