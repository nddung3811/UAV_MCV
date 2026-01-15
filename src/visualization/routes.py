from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# Global plotting style (force light theme)
# =========================================================

plt.style.use("default")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",

    "axes.edgecolor": "0.3",
    "axes.labelcolor": "0.2",
    "xtick.color": "0.2",
    "ytick.color": "0.2",
    "text.color": "0.15",

    "grid.color": "0.85",
    "grid.alpha": 0.18,

    "font.size": 11,
})


# =========================================================
# Helpers
# =========================================================

def _as_2d(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    return arr if arr.ndim == 2 else arr.reshape(-1, 2)


# =========================================================
# 1) UAV ROUTE (OVERVIEW)
# =========================================================

def plot_uav_route(
    points: np.ndarray,
    route_1based: np.ndarray,
    depot: np.ndarray,
    save_path: str | None = None,
):
    """
    Plot overall UAV route.
    """
    P = _as_2d(points)
    depot_id = P.shape[0] + 1

    coords = [
        depot if int(idx) == depot_id else P[int(idx) - 1]
        for idx in route_1based
    ]
    path = np.vstack(coords)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.scatter(
        P[:, 0], P[:, 1],
        s=24,
        facecolors="0.85",
        edgecolors="0.55",
        label="Reconnaissance area",
    )

    ax.plot(
        path[:, 0], path[:, 1],
        lw=1.3,
        color="0.4",
        label="UAV route",
    )

    ax.scatter(
        depot[0], depot[1],
        s=50,
        color="#6fa8dc",
        edgecolors="0.2",
        label="Base",
    )

    ax.set_title("UAV route", fontsize=14, weight="bold")
    ax.axis("equal")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="lower center", ncol=3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300) if save_path else plt.show()
    plt.close()


# =========================================================
# 2) MCV ROUTE (OVERVIEW)
# =========================================================

def plot_mcv_route(
    points: np.ndarray,
    route_1based: np.ndarray,
    charging_path: list[np.ndarray] | np.ndarray,
    depot: np.ndarray,
    save_path: str | None = None,
):
    """
    Plot MCV route with charging points over UAV route.
    """
    if (
        charging_path is None
        or len(charging_path) == 0
        or charging_path[0] is None
        or len(charging_path[0]) == 0
    ):
        return

    P = _as_2d(points)
    depot_id = P.shape[0] + 1

    coords = [
        depot if int(idx) == depot_id else P[int(idx) - 1]
        for idx in route_1based
    ]
    uav_path = np.vstack(coords)

    mcv_path = np.asarray(charging_path[0], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    ax.plot(
        uav_path[:, 0], uav_path[:, 1],
        lw=1.0,
        color="0.85",
        zorder=1,
    )

    ax.scatter(
        P[:, 0], P[:, 1],
        s=22,
        facecolors="0.85",
        edgecolors="0.55",
        zorder=2,
    )

    ax.plot(
        mcv_path[:, 0], mcv_path[:, 1],
        lw=1.8,
        color="#d62728",
        label="MCV route",
        zorder=4,
    )

    ax.scatter(
        mcv_path[:, 0], mcv_path[:, 1],
        s=30,
        color="#d62728",
        edgecolors="0.2",
        zorder=5,
        label="Charging points",
    )

    ax.scatter(
        depot[0], depot[1],
        s=50,
        color="#6fa8dc",
        edgecolors="0.2",
        zorder=6,
    )

    ax.set_title("MCV routes & charging points", fontsize=14, weight="bold")
    ax.axis("equal")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="lower center", ncol=2)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300) if save_path else plt.show()
    plt.close()

def plot_uav_routes_detailed(
    points: np.ndarray,
    route_1based: np.ndarray,
    depot: np.ndarray,
    assignment: dict,
    title: str = "UAV routes (detail)",
    save_path: str | None = None,
):
    """
    Plot detailed UAV routes per UAV (defensive).
    """
    n_points = points.shape[0]
    depot_id = n_points + 1

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(points[:, 0], points[:, 1], s=22, c="0.8", edgecolors="0.5")
    ax.scatter(depot[0], depot[1], marker="*", s=260, color="#6fa8dc")

    def _coord(node: int):
        return depot if node == depot_id else points[node - 1]

    uav_routes = assignment.get("uav_routes_nodes", {})

    if uav_routes:
        for uid, nodes in uav_routes.items():
            if not nodes:
                continue
            xs, ys = zip(*(_coord(int(n)) for n in nodes))
            ax.plot(xs, ys, lw=1.4, label=f"UAV{uid}")

        ax.legend()

    ax.set_title(title, fontsize=14, weight="bold")
    ax.axis("equal")
    ax.grid(True, alpha=0.18)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300) if save_path else plt.show()
    plt.close()
def plot_mcv_charging_detailed(
    points: np.ndarray,
    route_1based: np.ndarray,
    depot: np.ndarray,
    assignment: dict,
    title: str,
    save_path: str | None = None,
):
    """
    Plot detailed MCV charging routes (defensive).
    """
    n_points = points.shape[0]
    depot_id = n_points + 1

    fig, ax = plt.subplots(figsize=(8, 6))

    coords = [
        depot if int(idx) == depot_id else points[int(idx) - 1]
        for idx in route_1based
    ]
    uav_path = np.vstack(coords)

    ax.plot(uav_path[:, 0], uav_path[:, 1], lw=1.0, color="0.88", label="UAV route")

    ax.scatter(points[:, 0], points[:, 1], s=22, c="0.8", edgecolors="0.5")
    ax.scatter(depot[0], depot[1], marker="*", s=260, color="#6fa8dc")

    colors = plt.cm.tab10.colors

    mcv_points = assignment.get("mcv_charge_points_sorted", {})

    for mcv_id, pts in mcv_points.items():
        if not pts:
            continue

        pts = np.asarray(pts, dtype=float)
        color = colors[mcv_id % len(colors)]

        xs = [depot[0]] + pts[:, 0].tolist() + [depot[0]]
        ys = [depot[1]] + pts[:, 1].tolist() + [depot[1]]

        ax.plot(xs, ys, lw=1.8, color=color, label=f"MCV{mcv_id}")
        ax.scatter(pts[:, 0], pts[:, 1], s=30, color=color, edgecolors="0.2")

    ax.set_title(title, fontsize=14, weight="bold")
    ax.axis("equal")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="lower center", ncol=3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300) if save_path else plt.show()
    plt.close()
