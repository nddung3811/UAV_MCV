from __future__ import annotations
import numpy as np


def genetic_operator(
    parents: np.ndarray,
    n_routes: int,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    rng: np.random.Generator,
    crossover_prob: float = 1.0,
    crossover_dist: float = 20.0,
    mutation_prob: float = 1.0,
    mutation_dist: float = 20.0,
) -> np.ndarray:
    """
    Genetic operator for mixed-integer chromosomes:
    - Block 1   : discrete genes (one-point crossover + random mutation)
    - Block 2&3 : continuous genes (SBX crossover + polynomial mutation)

    Parameters
    ----------
    parents : (2N, D) ndarray
        Parent population.
    n_routes : int
        Number of discrete route genes.
    lower_bound, upper_bound : (D,) ndarray
        Variable bounds.
    rng : np.random.Generator
        Random generator.
    """

    half = parents.shape[0] // 2
    parent_a = parents[:half]
    parent_b = parents[half:2 * half]

    idx_discrete = slice(0, n_routes)
    idx_continuous = slice(n_routes, 3 * n_routes)

    # ===== Block 1: discrete genes =====
    gene_a = parent_a[:, idx_discrete].copy()
    gene_b = parent_b[:, idx_discrete].copy()
    num_ind, num_gene = gene_a.shape

    offspring_a = gene_a.copy()
    offspring_b = gene_b.copy()

    if num_ind > 0 and num_gene > 0:
        cut_points = rng.integers(1, num_gene + 1, size=num_ind)
        do_crossover = rng.random(num_ind) <= crossover_prob

        for i in range(num_ind):
            if do_crossover[i]:
                cut = cut_points[i] - 1
                offspring_a[i, cut:] = gene_b[i, cut:]
                offspring_b[i, cut:] = gene_a[i, cut:]

        lo = int(round(lower_bound[idx_discrete][0]))
        hi = int(round(upper_bound[idx_discrete][0]))

        mutation_mask = rng.random((2 * num_ind, num_gene)) < (mutation_prob / max(num_gene, 1))
        random_values = rng.integers(lo, hi + 1, size=(2 * num_ind, num_gene))

        discrete_offspring = np.vstack([offspring_a, offspring_b]).astype(float)
        discrete_offspring[mutation_mask] = random_values[mutation_mask]
    else:
        discrete_offspring = np.vstack([gene_a, gene_b]).astype(float)

    # ===== Block 2 & 3: continuous genes =====
    cont_a = parent_a[:, idx_continuous].copy()
    cont_b = parent_b[:, idx_continuous].copy()
    num_ind2, num_gene2 = cont_a.shape

    if num_ind2 > 0 and num_gene2 > 0:
        rand = rng.random((num_ind2, num_gene2))
        beta = np.empty_like(rand)

        beta[rand <= 0.5] = (2 * rand[rand <= 0.5]) ** (1.0 / (crossover_dist + 1))
        beta[rand > 0.5] = (2 - 2 * rand[rand > 0.5]) ** (-1.0 / (crossover_dist + 1))
        beta *= (-1) ** rng.integers(0, 2, size=(num_ind2, num_gene2))

        gene_mask = rng.random((num_ind2, num_gene2)) < 0.5
        beta[~gene_mask] = 1.0
        do_crossover = rng.random(num_ind2) <= crossover_prob
        beta[~do_crossover, :] = 1.0

        child_top = (cont_a + cont_b) / 2 + beta * (cont_a - cont_b) / 2
        child_bottom = (cont_a + cont_b) / 2 - beta * (cont_a - cont_b) / 2
        continuous_offspring = np.vstack([child_top, child_bottom])

        lb = np.broadcast_to(lower_bound[idx_continuous], continuous_offspring.shape)
        ub = np.broadcast_to(upper_bound[idx_continuous], continuous_offspring.shape)
        continuous_offspring = np.clip(continuous_offspring, lb, ub)

        mutation_mask = rng.random(continuous_offspring.shape) < (mutation_prob / max(num_gene2, 1))
        rand2 = rng.random(continuous_offspring.shape)

        mask_low = mutation_mask & (rand2 <= 0.5)
        mask_high = mutation_mask & (rand2 > 0.5)

        if np.any(mask_low):
            continuous_offspring[mask_low] += (ub[mask_low] - lb[mask_low]) * (
                (2 * rand2[mask_low] +
                 (1 - 2 * rand2[mask_low]) *
                 (1 - (continuous_offspring[mask_low] - lb[mask_low]) /
                  (ub[mask_low] - lb[mask_low])) ** (mutation_dist + 1))
                ** (1.0 / (mutation_dist + 1)) - 1.0
            )

        if np.any(mask_high):
            continuous_offspring[mask_high] += (ub[mask_high] - lb[mask_high]) * (
                1.0 -
                (2 * (1 - rand2[mask_high]) +
                 2 * (rand2[mask_high] - 0.5) *
                 (1 - (ub[mask_high] - continuous_offspring[mask_high]) /
                  (ub[mask_high] - lb[mask_high])) ** (mutation_dist + 1))
                ** (1.0 / (mutation_dist + 1))
            )

        continuous_offspring = np.clip(continuous_offspring, lb, ub)
    else:
        continuous_offspring = np.vstack([cont_a, cont_b]).astype(float)

    return np.hstack([discrete_offspring, continuous_offspring])
