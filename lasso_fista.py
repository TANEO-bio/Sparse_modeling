import numpy as np
import copy
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Lasso_Fista:
    def __init__(self, alpha=1.0, max_iter=5000,  tol=1e-05, normalize=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.normalize=normalize

    def soft_threashold(self, y, alpha):
        return np.sign(y) * np.maximum(np.abs(y) - alpha, 0.0)

    def supermum_eigen(self, A):
        return np.max(np.sum(np.abs(A), axis=0))

    def update(self, x, A, y, alpha, rho, beta, w):
        x_new = self.soft_threashold(w + (A.T @ (y - A @ w)) / rho, alpha / rho)
        beta_new = (1 + np.sqrt(1 + 4 * beta ** 2)) / 2
        w_new = x_new + ((beta - 1) / beta_new) * (x_new - x)
        return x_new, beta_new, w_new

    def fit(self, A, y):
        x = A.T @ y
        w = np.zeros(A.shape[1])
        n_samples = A.shape[0]
        beta = 0

        # 標準化
        if self.normalize==True:
            sscaler = StandardScaler()
            sscaler.fit(A)
            A = sscaler.transform(A)

        # rho, alpha
        rho = self.supermum_eigen(A.T @ A)
        alpha = self.alpha * n_samples

        for _ in range(self.max_iter):
            x_new, beta, w = self.update(x, A, y, alpha, rho, beta, w)
            if (np.abs(x - x_new) < self.tol).all():
                return x_new
            x = x_new
        raise ValueError('Not converged.')


if __name__ == "__main__":
    # データをロード
    Boston = load_boston()
    X = Boston.data
    y = Boston.target

    # 学習用・検証用にデータを分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    model = Lasso_Fista(normalize=True)
    res = model.fit(X_train, y_train)

    print(res)
