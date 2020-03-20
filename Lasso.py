import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LassoADMM:
    def __init__(self, rho=1.0, alpha=0.5, max_iter=100,  normalize=True):
        self.rho = rho
        self.alpha = alpha
        self.max_iter = max_iter
        self.normalize=normalize
        self.coef = None

    def soft_threshold(self, y, alpha):
        return np.sign(y) * np.maximum(np.abs(y) - alpha, 0.0)

    def update(self, x, y, z, A, b, inv_matrix, N, M):
        x = np.dot(inv_matrix, np.dot(A.T, b) / N + self.rho * z - y)
        z = self.soft_threshold(x + y / self.rho, self.alpha / self.rho)
        y += self.rho * (x - z)
        return x, y, z


    def fit(self, A, b):
        N = A.shape[0]
        M = A.shape[1]
        inv_matrix = np.linalg.inv(np.dot(A.T, A) / N + self.rho * np.identity(M))

        x = np.dot(A.T, b) / N
        y = np.zeros(M)
        z = x.copy()

        x_new = x.copy()

        for iteration in range(self.max_iter):
            x, y, z = self.update(x, y, z, A, b, inv_matrix, N, M)

        self.coef = z
        return self


if __name__ == "__main__":
    # データをロード
    Boston = load_boston()
    X = Boston.data
    y = Boston.target

    # 学習用・検証用にデータを分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    model = Lasso_ADMM()
    res = model.fit(X_train, y_train)

    print(res.coef)
