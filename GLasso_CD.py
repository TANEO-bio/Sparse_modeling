import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from Lasso import LassoADMM
from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso


class GraphicalLassoCD(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.precision = None
        self.covariance = None

    def fit(self, A, max_iter=30):
        sample_cov = np.cov(A.T, bias=1)
        n_params = len(sample_cov)
        sigma = np.copy(sample_cov)
        omega = np.zeros_like(sigma)
        indices = np.arange(n_params)

        for _ in range(max_iter):
            for i in range(n_params):
                sigma_hat = np.delete(sigma, i, axis=0)
                sigma_hat = np.delete(sigma_hat, i, axis=1)
                w, v = np.linalg.eigh(sigma_hat)
                sigma_root = v @ np.diag(np.sqrt(w)) @ v.T
                sigma_root_inv = v @ np.diag(np.reciprocal(np.sqrt(w))) @ v.T
                s1 = np.delete(sample_cov[i, :], i)

                model = LassoADMM(alpha=0.025)
                res = model.fit(sigma_root, sigma_root_inv @ s1.T)
                beta = res.coef

                sigma1 = sigma_hat @ beta
                omega_ = 1 / (sigma[i, i] - sigma1 @ beta.T)
                omega1 = -1 * omega_ * beta

                sigma[:, i][indices != i] = sigma1
                sigma[i, :][indices != i] = sigma1
                omega[i][i] = omega_
                omega[:, i][indices != i] = omega1
                omega[i, :][indices != i] = omega1

        self.precision = omega
        self.covariance = sigma
        return self


if __name__ == "__main__":
    # データをロード
    Boston = load_boston()
    X = sp.stats.zscore(Boston.data)

    # 学習用・検証用にデータを分割
    model = GraphicalLassoCD()
    res = model.fit(X)

    plt.imshow(res.covariance, interpolation='nearest', vmin=0, vmax=1, cmap='jet')
    plt.colorbar()
    plt.show()
    print(res.covariance)
    print(res.precision)
