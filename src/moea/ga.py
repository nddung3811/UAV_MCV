from __future__ import annotations
import numpy as np

def GA(Parent: np.ndarray, n_route: int, lower: np.ndarray, upper: np.ndarray,
       proC: float = 1.0, disC: float = 20.0, proM: float = 1.0, disM: float = 20.0) -> np.ndarray:
    """
    Port của GA.m (đã sửa mask để tránh IndexError):
      - Parent: (M, 3*n_route)
      - block1 (int-like)  : one-point crossover + bitwise mutation (re-sample integer)
      - block2+3 (float)   : SBX + polynomial mutation, clip theo [lower, upper]
    """
    # Chia nửa bố mẹ
    half = Parent.shape[0] // 2
    P1 = Parent[:half]
    P2 = Parent[half:2 * half]

    b1   = slice(0, n_route)           # block1
    b2b3 = slice(n_route, 3 * n_route) # block2+3

    # ===================== BLOCK 1 (rời rạc): one-point crossover + bitwise mutation =====================
    A1 = P1[:, b1].copy()   # (N, D)
    A2 = P2[:, b1].copy()
    N, D = A1.shape
    Off1 = A1.copy()
    Off2 = A2.copy()

    if D > 0 and N > 0:
        # one-point crossover theo từng cá thể, xác suất proC
        cuts = np.random.randint(1, D + 1, size=N)             # vị trí cắt 1..D
        do_crossover = (np.random.rand(N) <= proC)              # mask theo hàng
        for r in range(N):
            if do_crossover[r]:
                cut = cuts[r]
                # đổi đoạn [cut..end] giữa 2 cha mẹ
                Off1[r, cut - 1:] = A2[r, cut - 1:]
                Off2[r, cut - 1:] = A1[r, cut - 1:]

        # Bitwise mutation: mỗi gene có xác suất proM/D được thay ngẫu nhiên trong [lo1..hi1]
        lo1 = int(round(lower[b1][0]))
        hi1 = int(round(upper[b1][0]))
        if D > 0:
            site = (np.random.rand(2 * N, D) < (proM / max(D, 1)))
            rand_vals = np.random.randint(lo1, hi1 + 1, size=(2 * N, D))
            B1 = np.vstack([Off1, Off2]).astype(float)
            B1[site] = rand_vals[site]
        else:
            B1 = np.vstack([Off1, Off2]).astype(float)
    else:
        B1 = np.vstack([A1, A2]).astype(float)  # trường hợp N=0 hoặc D=0

    # ===================== BLOCK 2+3 (liên tục): SBX + polynomial mutation =====================
    A3 = P1[:, b2b3].copy()  # (N2, D2)
    A4 = P2[:, b2b3].copy()
    N2, D2 = A3.shape

    if D2 > 0 and N2 > 0:
        # SBX
        mu = np.random.rand(N2, D2)
        beta = np.empty_like(mu)
        beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1.0 / (disC + 1.0))
        beta[mu > 0.5]  = (2 - 2 * mu[mu > 0.5]) ** (-1.0 / (disC + 1.0))
        beta *= (-1) ** np.random.randint(0, 2, size=(N2, D2))

        # Áp dụng xác suất crossover theo HÀNG (nếu hàng không lai, beta=1 trên cả hàng)
        do_crossover23 = (np.random.rand(N2) <= proC)
        if np.any(~do_crossover23):
            beta[~do_crossover23, :] = 1.0

        C_top = (A3 + A4) / 2 + beta * (A3 - A4) / 2
        C_bot = (A3 + A4) / 2 - beta * (A3 - A4) / 2
        B23 = np.vstack([C_top, C_bot])

        # Polynomial mutation
        lower23 = np.broadcast_to(lower[b2b3], (2 * N2, D2))
        upper23 = np.broadcast_to(upper[b2b3], (2 * N2, D2))
        B23 = np.minimum(np.maximum(B23, lower23), upper23)

        site = (np.random.rand(2 * N2, D2) < (proM / max(D2, 1)))
        mu2  = np.random.rand(2 * N2, D2)

        # nhánh mu<=0.5
        temp1 = site & (mu2 <= 0.5)
        if np.any(temp1):
            B23[temp1] = B23[temp1] + (upper23[temp1] - lower23[temp1]) * (
                (2 * mu2[temp1] + (1 - 2 * mu2[temp1]) *
                 (1 - (B23[temp1] - lower23[temp1]) / (upper23[temp1] - lower23[temp1])) ** (disM + 1)) ** (1.0 / (disM + 1)) - 1.0
            )

        # nhánh mu>0.5
        temp2 = site & (mu2 > 0.5)
        if np.any(temp2):
            B23[temp2] = B23[temp2] + (upper23[temp2] - lower23[temp2]) * (
                1.0 - (2 * (1 - mu2[temp2]) + 2 * (mu2[temp2] - 0.5) *
                       (1 - (upper23[temp2] - B23[temp2]) / (upper23[temp2] - lower23[temp2])) ** (disM + 1)) ** (1.0 / (disM + 1))
            )

        # clip lần nữa
        B23 = np.minimum(np.maximum(B23, lower23), upper23)
    else:
        B23 = np.vstack([A3, A4]).astype(float)  # trường hợp N2=0 hoặc D2=0

    # ===================== GHÉP 3 khối =====================
    Offspring = np.hstack([B1, B23])
    return Offspring
