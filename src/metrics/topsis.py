from __future__ import annotations
import numpy as np

def topsis_generic(matrix: np.ndarray, weights: np.ndarray, benefit_mask: np.ndarray) -> np.ndarray:
    """
    Bản TOPSIS tổng quát (đã gửi trước) – giữ lại nếu bạn cần dùng sau.
    Trả về: scores (cao hơn là tốt).
    """
    X = np.asarray(matrix, dtype=float)
    n, m = X.shape
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = w / (w.sum() + 1e-12)
    bmask = np.asarray(benefit_mask, dtype=bool).reshape(-1)

    denom = np.linalg.norm(X, axis=0, keepdims=True) + 1e-12
    V = (X / denom) * w
    ideal_best  = np.where(bmask,  V.max(axis=0), V.min(axis=0))
    ideal_worst = np.where(bmask,  V.min(axis=0), V.max(axis=0))
    d_best  = np.linalg.norm(V - ideal_best, axis=1)
    d_worst = np.linalg.norm(V - ideal_worst, axis=1)
    return d_worst / (d_best + d_worst + 1e-12)

def topsis_matlab(obj: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Port đúng theo MATLAB:
    function [obj1,index1]=TOPSIS(obj)
      n = size(obj,1);
      max1 = max(obj,[],1);
      nor_obj = repmat(max1,n,1) - obj;
      nor_obj = nor_obj./repmat(sqrt(sum(nor_obj.^2,1)),n,1);
      max_nor_obj = max(nor_obj,[],1);
      min_nor_obj = min(nor_obj,[],1);
      D_max = sqrt(sum((repmat(max_nor_obj,n,1)-nor_obj).^2,2));
      D_min = sqrt(sum((repmat(min_nor_obj,n,1)-nor_obj).^2,2));
      s = D_min./(D_max+D_min);
      [~,index1] = max(s);
      obj1 = obj(index1,:);
    end
    Trả về:
      obj1   : hàng obj được chọn (shape (m,))
      index1 : chỉ số (int)
      s      : vector điểm TOPSIS (shape (n,))
    """
    X = np.asarray(obj, dtype=float)
    if X.ndim != 2:
        raise ValueError("obj must be 2D array")
    n, m = X.shape
    # nor_obj = (max_col - X), rồi chuẩn hoá theo chuẩn 2 từng cột
    max1 = X.max(axis=0, keepdims=True)               # 1×m
    nor_obj = (max1 - X)                               # n×m
    denom = np.sqrt((nor_obj ** 2).sum(axis=0, keepdims=True)) + 1e-12
    nor_obj = nor_obj / denom                          # n×m

    max_nor = nor_obj.max(axis=0, keepdims=True)       # 1×m
    min_nor = nor_obj.min(axis=0, keepdims=True)       # 1×m

    D_max = np.sqrt(((max_nor - nor_obj) ** 2).sum(axis=1))  # n
    D_min = np.sqrt(((min_nor - nor_obj) ** 2).sum(axis=1))  # n

    s = D_min / (D_max + D_min + 1e-12)                # n
    index1 = int(np.argmax(s))
    obj1 = X[index1].copy()
    return obj1, index1, s
