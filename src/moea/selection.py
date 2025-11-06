from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist

def tournament_selection(k: int, pop_size: int, fitness: np.ndarray) -> np.ndarray:
    """
    Port TournamentSelection.m
    - rank theo fitness (nhỏ tốt)
    - chọn k ứng viên ngẫu nhiên, lấy tốt nhất
    Trả về: chỉ số (pop_size,)
    """
    n = len(fitness)
    order = np.argsort(fitness)
    rank = np.empty_like(order)
    rank[order] = np.arange(n)

    Parents = np.random.randint(0, n, size=(k, pop_size))
    best_rows = np.argmin(rank[Parents], axis=0)
    idx = Parents[best_rows, np.arange(pop_size)]
    return idx.astype(int)


def _truncation(PopObj: np.ndarray, K: int) -> np.ndarray:
    """
    Port Truncation.m
    Chọn K chỉ số để loại bỏ (True ở các vị trí bị xoá)
    """
    Distance = cdist(PopObj, PopObj)
    np.fill_diagonal(Distance, np.inf)
    Del = np.zeros(PopObj.shape[0], dtype=bool)
    while Del.sum() < K:
        Remain = np.where(~Del)[0]
        Temp = np.sort(Distance[Remain][:, Remain], axis=1)
        # tìm điểm có khoảng cách lân cận nhỏ nhất -> xoá
        Rank = np.lexsort(Temp.T)  # sortrows tương đương
        Del[Remain[Rank[0]]] = True
    return Del


def environmental_selection(Population: np.ndarray,
                            Fitness: np.ndarray,
                            N: int,
                            obj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Port EnvironmentalSelection.m
    - Lấy tất cả nghiệm Fitness < 1
    - Nếu thiếu -> bù theo fitness tốt nhất
    - Nếu thừa -> dùng truncation theo obj
    Trả về: (Population_next, Fitness_next)
    """
    Next = Fitness < 1.0
    cnt = int(Next.sum())
    if cnt < N:
        order = np.argsort(Fitness)
        Next[order[:N]] = True
    elif cnt > N:
        # áp dụng truncation trên tập Next
        Temp_idx = np.where(Next)[0]
        Del_local = _truncation(obj[Temp_idx], cnt - N)
        Next[Temp_idx[Del_local]] = False

    PopN = Population[Next]
    FitN = Fitness[Next]
    # sort theo fitness
    order = np.argsort(FitN)
    return PopN[order], FitN[order]
