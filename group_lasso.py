import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer


class Lasso_CoordinateDescent(object):
    def __init__(self, alpha=1.0, max_iter=1000, normalize=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.normalize = normalize
        self.coef = None

    def soft_threashold(self, y, alpha):
        return np.sign(y) * np.maximum(np.abs(y) - alpha, 0.0)

    def supermum_eigen(self, A):
        return np.max(np.sum(np.abs(A), axis=0))

    def update(self, w, A, y, rho, alpha, group_ids, group_id_sum):
        # wの微分
        threshold = alpha / rho
        diff = w + (A.T @ (y - A @ w)) / rho

        # groupにおける回帰係数のL2ノルム
        group_norm = np.sqrt(diff @ np.diag(diff) @ group_ids)

        # 閾値未満のL2ノルムを倍率に変換、閾値未満を0に
        multiplier = np.ones(len(group_norm)) - \
            np.full_like(group_norm, threshold) / group_norm
        multiplier = np.maximum(0, multiplier)

        # 回帰係数 * グループの倍率、wの更新
        group_w = multiplier @ group_ids.T
        w_new = diff * group_w
        return w_new

    def fit(self, A, y, group_ids):
        # 標準化
        if self.normalize == True:
            sscaler = StandardScaler()
            sscaler.fit(A)
            A = sscaler.transform(A)

        # 初期値、ハイパラ
        n_samples = A.shape[0]
        w = np.zeros(A.shape[1])
        rho = self.supermum_eigen(A.T @ A)
        alpha = self.alpha * n_samples

        # group-ids -> one-hot vector
        group_ids = LabelBinarizer().fit_transform(group_ids)
        group_ids = group_ids.astype(np.float64)
        group_id_sum = np.sum(group_ids, axis=0)

        for it in range(self.max_iter):
            w = self.update(w, A, y, rho, alpha, group_ids, group_id_sum)
        self.coef = w
        return self
