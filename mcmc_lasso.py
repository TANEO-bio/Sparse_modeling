import pystan
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# データをロード
Boston = load_boston()
X = Boston.data
y = Boston.target

# 学習用・検証用にデータを分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
stan_data = {'N': X_train.shape[0], 'M': X_train.shape[1], 'X': X_train, 'y': y_train, 'Lambda': 0.2}

# Model作成
model = """data {
        int<lower=0> N;
        int<lower=0>M;
        matrix[N, M] X;
        vector[N] y;
        real<lower=0> Lambda;
}

parameters {
        real beta_0;
        vector[M] beta;
}

transformed parameters {
   real<lower=0> squared_error;
   squared_error <- dot_self(y - X * beta - beta_0);
}

model {
    increment_log_prob(-squared_error);
    for (m in 1:M)
      increment_log_prob(-Lambda * abs(beta[m]));
}"""

# StanでModelをコンパイル・推定
stm = pystan.StanModel(model_code=model)
fit = stm.sampling(data=stan_data, iter=3000, chains=3,thin=1)


# 参考文献
# http://tekenuko.hatenablog.com/entry/2017/10/14/150405
# http://statmodeling.hatenablog.com/entry/bayesian-lass
