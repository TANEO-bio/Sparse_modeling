import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.covariance import GraphicalLasso


# ---product---

class GraphicalLassoADMM(object):

    def __init__(self, alpha=0.6, rho=1.0, max_iter=100):
        self.alpha = alpha
        self.rho = rho
        self.max_iter = max_iter
        self.cov = None
        self.precision = None


    def update_X(self, Y, Z, cov):
        C = Y - Z - (1 / self.alpha) * cov
        w, v = np.linalg.eigh(C)
        return v @ ((1/2) * np.diag(w + np.sqrt(w ** 2 + 4 / self.alpha))) @ v.T

    def soft_threshold(self, y, rho):
        return np.sign(y) * np.maximum(np.abs(y) - (self.alpha / self.rho), 0.0)

    def fit(self, A):
        # initialize variable
        cov = np.cov(A.T, bias=1)
        print(cov)
        Y = np.linalg.inv(np.copy(cov))
        Z = np.zeros_like(Y)

        # optimize covariance
        for _ in range(self.max_iter):
            X = self.update_X(Y, Z, cov)
            Y = self.soft_threshold(X + Z, self.alpha / self.rho)
            Z = Z + self.alpha * (X - Y)

        self.cov = np.linalg.inv(Y)
        self.precision = Y
        return self

if __name__ == "__main__":
    A = load_boston().data
    A = sp.stats.zscore(A, axis=0)

    # ---sklearn---
    model = GraphicalLasso(alpha=0.4,verbose=True)
    model.fit(A)

    cov = np.cov(A.T)
    cov_ = model.covariance_
    pre_ = model.precision_
    model = GraphicalLassoADMM()
    res = model.fit(A)
    #print(res.precision)
    #print(cov_)

    # 普通の共分散行列
    plt.imshow(cov,interpolation='nearest',vmin=0,vmax=1,cmap='jet')
    plt.colorbar()
    plt.figure()

    # sklearnのglasso
    plt.imshow(cov_,interpolation='nearest',vmin=0,vmax=1,cmap='jet')
    plt.colorbar()
    plt.figure()

    # 実装したもの
    plt.imshow(res.cov,interpolation='nearest',vmin=0,vmax=1,cmap='jet')
    plt.colorbar()
    plt.show()
