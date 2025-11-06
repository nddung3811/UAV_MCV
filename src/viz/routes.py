# src/viz/routes.py
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


def plot_uav_route(point: np.ndarray, route_1based: np.ndarray, start_point: np.ndarray):
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
    ax.scatter(P[:, 0], P[:, 1], s=22, facecolors=(0.8, 0.8, 0.8), edgecolors=(0.6, 0.6, 0.6), label="Reconnaissance area")
    ax.plot(C[:, 0], C[:, 1], "-", lw=1.2, label="UAVs routes")
    ax.scatter(start_point[0], start_point[1], s=40, color="#69a7e6", edgecolors="k", label="Base")
    ax.set_title("UAV route", fontsize=14, weight="bold")
    ax.axis("equal"); ax.grid(True, alpha=0.25); ax.legend(loc="lower center", ncol=3)
    plt.tight_layout(); plt.show()


def plot_mcv_route(point: np.ndarray, route_1based: np.ndarray, point_cha_sort, start_point: np.ndarray):
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
    ax.plot(C[:, 0], C[:, 1], "-", lw=0.9, color=(0.7, 0.7, 0.7), alpha=0.9, zorder=1)
    ax.scatter(P[:, 0], P[:, 1], s=20, facecolors=(0.75, 0.75, 0.75), edgecolors=(0.45, 0.45, 0.45))
    ax.plot(pcs[:, 0], pcs[:, 1], "-r", lw=1.6, zorder=3, label="MCVs routes")
    ax.scatter(pcs[:, 0], pcs[:, 1], s=24, c="r", edgecolors="k", zorder=4, label="charging points")
    ax.scatter(start_point[0], start_point[1], s=44, color="#69a7e6", edgecolors="k", zorder=5)

    ax.set_title("MCV routes & charging points", fontsize=14, weight="bold")
    ax.axis("equal"); ax.grid(True, alpha=0.25); ax.legend(loc="lower center", ncol=2)
    plt.tight_layout(); plt.show()
