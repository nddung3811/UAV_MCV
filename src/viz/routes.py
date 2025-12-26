from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

try:
    plt.style.use("dark_background")
except Exception:
    pass


def _ensure2d(a):
    a = np.asarray(a, dtype=float)
    return a if a.ndim == 2 else a.reshape(-1, 2)


# ============================================================
# 1) UAV ROUTE (TỔNG)
# ============================================================
def plot_uav_route(point: np.ndarray,
                   route_1based: np.ndarray,
                   start_point: np.ndarray,
                   save_path: str | None = None):

    P = _ensure2d(point)
    nn = P.shape[0] + 1

    coords = []
    for idx in route_1based:
        if int(idx) == nn:
            coords.append(np.asarray(start_point, float))
        else:
            coords.append(P[int(idx) - 1])
    C = np.vstack(coords)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.scatter(
        P[:, 0], P[:, 1],
        s=22,
        facecolors=(0.8, 0.8, 0.8),
        edgecolors=(0.6, 0.6, 0.6),
        label="Reconnaissance area"
    )

    ax.plot(C[:, 0], C[:, 1], "-", lw=1.2, label="UAVs routes")

    ax.scatter(
        start_point[0], start_point[1],
        s=40, color="#69a7e6", edgecolors="k", label="Base"
    )

    ax.set_title("UAV route", fontsize=14, weight="bold")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower center", ncol=3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


# ============================================================
# 2) MCV ROUTE (TỔNG)
# ============================================================
def plot_mcv_route(point: np.ndarray,
                   route_1based: np.ndarray,
                   point_cha_sort,
                   start_point: np.ndarray,
                   save_path: str | None = None):

    if not point_cha_sort or point_cha_sort[0] is None or len(point_cha_sort[0]) < 2:
        print("[plot_mcv_route] Không có điểm sạc → không vẽ MCV.")
        return

    P = _ensure2d(point)
    nn = P.shape[0] + 1

    coords = []
    for idx in route_1based:
        if int(idx) == nn:
            coords.append(np.asarray(start_point, float))
        else:
            coords.append(P[int(idx) - 1])
    C = np.vstack(coords)

    pcs = np.asarray(point_cha_sort[0], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    # UAV route nền
    ax.plot(
        C[:, 0], C[:, 1],
        "-",
        lw=0.9,
        color=(0.7, 0.7, 0.7),
        alpha=0.9,
        zorder=1
    )

    ax.scatter(
        P[:, 0], P[:, 1],
        s=20,
        facecolors=(0.75, 0.75, 0.75),
        edgecolors=(0.45, 0.45, 0.45)
    )

    # MCV route
    ax.plot(
        pcs[:, 0], pcs[:, 1],
        "-r",
        lw=1.6,
        zorder=3,
        label="MCVs routes"
    )

    ax.scatter(
        pcs[:, 0], pcs[:, 1],
        s=24,
        c="r",
        edgecolors="k",
        zorder=4,
        label="Charging points"
    )

    ax.scatter(
        start_point[0], start_point[1],
        s=44,
        color="#69a7e6",
        edgecolors="k",
        zorder=5
    )

    ax.set_title("MCV routes & charging points", fontsize=14, weight="bold")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower center", ncol=2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


# ============================================================
# 3) UAV ROUTES – CHI TIẾT THEO UAV
# ============================================================
def plot_uav_routes_detailed(point: np.ndarray,
                             route: np.ndarray,
                             start_point: np.ndarray,
                             assign: dict,
                             title: str = "UAV routes (by UAV id)",
                             save_path: str | None = None):

    n_point = point.shape[0]
    nn = n_point + 1

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(point[:, 0], point[:, 1])
    ax.scatter(start_point[0], start_point[1], marker="*", s=260)

    for i in range(n_point):
        ax.text(point[i, 0], point[i, 1], f"{i+1}")

    def xy(node):
        if node == nn:
            return float(start_point[0]), float(start_point[1])
        p = point[node - 1]
        return float(p[0]), float(p[1])

    for uid, nodes in assign["uav_routes_nodes"].items():
        if not nodes:
            continue

        xs, ys = [], []
        for node in nodes:
            x, y = xy(int(node))
            xs.append(x)
            ys.append(y)

        ax.plot(xs, ys, lw=1.2, alpha=0.9, label=f"UAV{uid}")

        for t, node in enumerate(nodes):
            x, y = xy(int(node))
            ax.text(x, y, f"U{uid}:{t}", fontsize=8)

    ax.set_title(title)
    ax.legend()
    ax.axis("equal")
    ax.grid(True, alpha=0.25)

    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_mcv_charging_detailed(point: np.ndarray,
                               route: np.ndarray,
                               start_point: np.ndarray,
                               assign: dict,
                               title: str = "MCV charging (detail)",
                               save_path: str | None = None):

    n_point = point.shape[0]
    nn = n_point + 1

    fig, ax = plt.subplots(figsize=(8, 6))

    # ====================================================
    # UAV ROUTE NỀN – GIỐNG plot_uav_route
    # ====================================================
    coords = []
    for idx in route:
        if int(idx) == nn:
            coords.append(np.asarray(start_point, float))
        else:
            coords.append(point[int(idx) - 1])
    C = np.vstack(coords)

    ax.plot(
        C[:, 0], C[:, 1],
        "-", lw=1.1,
        color=(0.85, 0.85, 0.85),
        alpha=0.9,
        zorder=0,
        label="UAV route"
    )

    # ====================================================
    # VẼ ĐIỂM
    # ====================================================
    ax.scatter(point[:, 0], point[:, 1],
               s=20,
               facecolors=(0.75, 0.75, 0.75),
               edgecolors=(0.45, 0.45, 0.45))

    ax.scatter(start_point[0], start_point[1],
               marker="*", s=260,
               color="#69a7e6", edgecolors="k")

    for i in range(n_point):
        ax.text(point[i, 0], point[i, 1], f"{i+1}")

    # ====================================================
    # MCV DETAIL – PHÂN BIỆT THEO MÀU
    # ====================================================
    colors = plt.cm.tab10.colors  # 10 màu phân biệt rõ

    for mcv_id, pts in assign["mcv_charge_points_sorted"].items():
        if not pts:
            continue

        pts = np.asarray(pts, float)
        color = colors[mcv_id % len(colors)]

        xs = [start_point[0]] + pts[:, 0].tolist() + [start_point[0]]
        ys = [start_point[1]] + pts[:, 1].tolist() + [start_point[1]]

        ax.plot(
            xs, ys,
            lw=1.6,
            color=color,
            alpha=0.95,
            label=f"MCV{mcv_id}",
            zorder=3
        )

        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=30,
            color=color,
            edgecolors="k",
            zorder=4
        )

    # ====================================================
    # FORMAT – GIỐNG UAV ROUTE / MCV ROUTE
    # ====================================================
    ax.set_title(title, fontsize=14, weight="bold")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower center", ncol=3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()
