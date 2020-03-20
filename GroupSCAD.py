import numpy as np
import scipy as sp
from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelBinarizer


class GroupSCAD(object):
    def __init__(self, rho=3.7, lambda_=1, max_iter=1000):
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

    def update_theta(self, w, theta_hat, group_ids):
        z = theta_hat - w @ np.sign(theta_hat) / self.rho
        z_group = np.sqrt(z ** 2 @ group_ids)

        z1 = np.where(np.abs(z_group @ group_ids.T) < self.lambda_, z, 0)
        multiplier = np.maximum(0, np.ones_like(z_group) - np.full_like(z_group, self.rho * self.lambda_ / (self.rho - 1)) / z_group) @ group_ids.T
        theta1 = z1 * multiplier

        z2 = np.where((np.abs(z_group @ group_ids.T) > self.lambda_) & (np.abs(z_group @ group_ids.T) < self.rho * self.lambda_), z, 0)
        multiplier = (self.rho - 1) / (self.rho - 2) * np.maximum(0, np.ones_like(z_group) - np.full_like(z_group, self.rho * self.lambda_ / (self.rho - 1)) / z_group) @ group_ids.T
        theta2 = z2 * multiplier

        z3 = np.where(np.abs(z_group @ group_ids.T) > self.rho * self.lambda_, 1, 0)
        theta3 = z * z3

        theta_new = theta1 + theta2 + theta3
        return theta_new

    def update_gamma(self, A, x, gamma, theta):
        return gamma - self.rho * (theta - x)

    def fit(self, A, y, group_ids):
        x = np.zeros(A.shape[1])
        theta = x.copy()
        gamma = x.copy()
        inv_matrix = np.linalg.inv(A.T @ A + self.rho)

        group_ids = LabelBinarizer().fit_transform(group_ids)
        group_ids = group_ids.astype(np.float64)

        for _ in range(self.max_iter):
            x = self.update_x(A, y, gamma, theta, inv_matrix)
            theta_hat = self.update_theta_hat(x, gamma)
            w = self.update_w(theta_hat)
            theta = self.update_theta(w, theta_hat, group_ids)
            gamma = self.update_gamma(A, x, gamma, theta)

        self.coef_ = theta
        return self


if __name__ == "__main__":
    A = load_boston().data
    y = load_boston().target
    A = sp.stats.zscore(A, axis=0)
    y = sp.stats.zscore(y)
    group_ids = np.array([0, 1, 1, 3, 3, 3, 4, 5, 6, 6, 7, 7, 7])

    # ---sklearnのlasso---
    model = Lasso(alpha=0.05)
    res = model.fit(A, y)
    print(res.coef_)

    # 実装したもの
    model = GroupSCAD()
    res = model.fit(A, y, group_ids)
    print(res.coef_)
