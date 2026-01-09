from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist


def tournament_selection(
    tournament_size: int,
    population_size: int,
    fitness: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Tournament selection (minimization).
    """
    order = np.argsort(fitness)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(fitness))

    candidates = rng.integers(0, len(fitness), size=(tournament_size, population_size))
    winners = np.argmin(rank[candidates], axis=0)
    return candidates[winners, np.arange(population_size)]


def _truncation(objectives: np.ndarray, remove_count: int) -> np.ndarray:
    """
    Distance-based truncation.
    """
    distance = cdist(objectives, objectives)
    np.fill_diagonal(distance, np.inf)

    removed = np.zeros(objectives.shape[0], dtype=bool)
    while removed.sum() < remove_count:
        remaining = np.where(~removed)[0]
        sorted_dist = np.sort(distance[remaining][:, remaining], axis=1)
        worst = np.lexsort(sorted_dist.T)[0]
        removed[remaining[worst]] = True

    return removed


def environmental_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    target_size: int,
    objectives: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Environmental selection using fitness threshold and truncation.
    """
    selected = fitness < 1.0
    count = selected.sum()

    if count < target_size:
        order = np.argsort(fitness)
        selected[order[:target_size]] = True
    elif count > target_size:
        idx = np.where(selected)[0]
        removed = _truncation(objectives[idx], count - target_size)
        selected[idx[removed]] = False

    next_pop = population[selected]
    next_fit = fitness[selected]

    order = np.argsort(next_fit)
    return next_pop[order], next_fit[order]
