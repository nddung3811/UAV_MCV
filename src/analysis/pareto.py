import numpy as np

def get_non_dominated_indices(obj: np.ndarray) -> np.ndarray:
    """
    Return indices of non-dominated points (minimization).
    """
    n = obj.shape[0]
    is_dom = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dom[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(obj[j] <= obj[i]) and np.any(obj[j] < obj[i]):
                is_dom[i] = True
                break

    return np.where(~is_dom)[0]
