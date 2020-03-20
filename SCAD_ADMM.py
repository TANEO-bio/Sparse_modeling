import numpy as np
import scipy as sp
from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso


class SCAD_ADMM(object):
    def __init__(self, rho=3.7, lambda_=1, max_iter=2):
        self.rho = rho
        self.lambda_ = lambda_
        self.coef_ = None
        self.max_iter = max_iter

    def update_x(self, A, y, gamma, theta, inv_matrix):
        return inv_matrix @ (A.T @ y + (self.rho * theta - gamma))

    def update_theta_hat(self, x, gamma):
        return x + gamma / self.rho

    def update_w(self, theta_hat):
        return np.where(np.abs(theta_hat) > self.lambda_, np.maximum(0, (np.full(len(theta_hat), self.rho * self.lambda_) - np.abs(theta_hat))) / (self.rho - 1), self.lambda_)

    def update_theta(self, w, theta_hat):
        return np.where(np.abs(theta_hat) <= w / self.rho, 0, theta_hat - w @ np.sign(theta_hat) / self.rho)

    def update_gamma(self, A, x, gamma, theta):
        return gamma - self.rho * (theta - x)

    def fit(self, A, y):
        x = np.zeros(A.shape[1])
        theta = x.copy()
        gamma = x.copy()
        inv_matrix = np.linalg.inv(A.T @ A + self.rho)

        for _ in range(self.max_iter):
            x = self.update_x(A, y, gamma, theta, inv_matrix)
            theta_hat = self.update_theta_hat(x, gamma)
            w = self.update_w(theta_hat)
            theta = self.update_theta(w, theta_hat)
            gamma = self.update_gamma(A, x, gamma, theta)

        self.coef_ = theta
        return self


if __name__ == "__main__":
    A = load_boston().data
    y = load_boston().target
    A = sp.stats.zscore(A, axis=0)
    y = sp.stats.zscore(y)

    # ---sklearnのlasso---
    model = Lasso(alpha=0.05)
    res = model.fit(A, y)

    # 実装したもの
    model = SCAD_ADMM()
    res = model.fit(A, y)
