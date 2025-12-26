from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("default")   # QUAN TRỌNG: reset mọi dark style trước đó

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",   # <<< THIẾU CÁI NÀY

    "axes.edgecolor": "0.3",
    "axes.labelcolor": "0.2",
    "xtick.color": "0.2",
    "ytick.color": "0.2",
    "text.color": "0.15",

    "grid.color": "0.85",
    "grid.alpha": 0.18,

    "font.size": 11,
})


def _ensure2d(a):
    a = np.asarray(a, dtype=float)
    return a if a.ndim == 2 else a.reshape(-1, 2)


# ============================================================
# 1) UAV ROUTE (TỔNG)
# ============================================================
def plot_uav_route(point, route_1based, start_point, save_path=None):

    P = _ensure2d(point)
    nn = P.shape[0] + 1

    coords = []
    for idx in route_1based:
        coords.append(start_point if int(idx) == nn else P[int(idx) - 1])
    C = np.vstack(coords)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.scatter(
        P[:, 0], P[:, 1],
        s=24,
        facecolors="0.85",
        edgecolors="0.55",
        label="Reconnaissance area"
    )

    ax.plot(
        C[:, 0], C[:, 1],
        lw=1.3,
        color="0.4",
        label="UAV route"
    )

    ax.scatter(
        start_point[0], start_point[1],
        s=50,
        color="#6fa8dc",
        edgecolors="0.2",
        label="Base"
    )

    ax.set_title("UAV route", fontsize=14, weight="bold")
    ax.axis("equal")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="lower center", ncol=3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300) if save_path else plt.show()
    plt.close()


# ============================================================
# 2) MCV ROUTE (TỔNG)
# ============================================================
def plot_mcv_route(point, route_1based, point_cha_sort, start_point, save_path=None):

    if not point_cha_sort or not point_cha_sort[0]:
        return

    P = _ensure2d(point)
    nn = P.shape[0] + 1

    coords = []
    for idx in route_1based:
        coords.append(start_point if int(idx) == nn else P[int(idx) - 1])
    C = np.vstack(coords)

    pcs = np.asarray(point_cha_sort[0], float)

    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    # UAV route nền
    ax.plot(
        C[:, 0], C[:, 1],
        lw=1.0,
        color="0.85",
        zorder=1
    )

    ax.scatter(
        P[:, 0], P[:, 1],
        s=22,
        facecolors="0.85",
        edgecolors="0.55",
        zorder=2
    )

    # MCV route
    ax.plot(
        pcs[:, 0], pcs[:, 1],
        lw=1.8,
        color="#d62728",
        label="MCV route",
        zorder=4
    )

    ax.scatter(
        pcs[:, 0], pcs[:, 1],
        s=30,
        color="#d62728",
        edgecolors="0.2",
        zorder=5,
        label="Charging points"
    )

    ax.scatter(
        start_point[0], start_point[1],
        s=50,
        color="#6fa8dc",
        edgecolors="0.2",
        zorder=6
    )

    ax.set_title("MCV routes & charging points", fontsize=14, weight="bold")
    ax.axis("equal")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="lower center", ncol=2)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300) if save_path else plt.show()
    plt.close()


# ============================================================
# 3) UAV ROUTES – CHI TIẾT
# ============================================================
def plot_uav_routes_detailed(point, route, start_point, assign, title, save_path=None):

    n_point = point.shape[0]
    nn = n_point + 1

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(point[:, 0], point[:, 1], s=22, c="0.8", edgecolors="0.5")
    ax.scatter(start_point[0], start_point[1], marker="*", s=260, color="#6fa8dc")

    def xy(node):
        return start_point if node == nn else point[node - 1]

    for uid, nodes in assign["uav_routes_nodes"].items():
        xs, ys = zip(*[xy(int(n)) for n in nodes])
        ax.plot(xs, ys, lw=1.4, label=f"UAV{uid}")

    ax.set_title(title, fontsize=14, weight="bold")
    ax.axis("equal")
    ax.grid(True, alpha=0.18)
    ax.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=300) if save_path else plt.show()
    plt.close()


# ============================================================
# 4) MCV CHARGING – DETAIL
# ============================================================
def plot_mcv_charging_detailed(point, route, start_point, assign, title, save_path=None):

    n_point = point.shape[0]
    nn = n_point + 1

    fig, ax = plt.subplots(figsize=(8, 6))

    # UAV route nền
    coords = []
    for idx in route:
        coords.append(start_point if int(idx) == nn else point[int(idx) - 1])
    C = np.vstack(coords)

    ax.plot(C[:, 0], C[:, 1], lw=1.0, color="0.88", label="UAV route")

    ax.scatter(point[:, 0], point[:, 1], s=22, c="0.8", edgecolors="0.5")
    ax.scatter(start_point[0], start_point[1], marker="*", s=260, color="#6fa8dc")

    colors = plt.cm.tab10.colors

    for mcv_id, pts in assign["mcv_charge_points_sorted"].items():
        if not pts:
            continue

        pts = np.asarray(pts)
        c = colors[mcv_id % len(colors)]

        xs = [start_point[0]] + pts[:, 0].tolist() + [start_point[0]]
        ys = [start_point[1]] + pts[:, 1].tolist() + [start_point[1]]

        ax.plot(xs, ys, lw=1.8, color=c, label=f"MCV{mcv_id}")
        ax.scatter(pts[:, 0], pts[:, 1], s=30, color=c, edgecolors="0.2")

    ax.set_title(title, fontsize=14, weight="bold")
    ax.axis("equal")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="lower center", ncol=3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300) if save_path else plt.show()
    plt.close()
