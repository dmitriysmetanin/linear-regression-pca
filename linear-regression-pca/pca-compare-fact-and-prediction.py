import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np

# Импорт файла и чтение данных
file = pd.read_csv('files/bikes_rent.csv')
X = file[['yr', 'season', 'temp']]
y = file[['cnt']]

def build_pca2_model(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    model = LinearRegression()
    model.fit(X_pca, y)

    return (model, X_pca, y)

# Строим PCA-модель с размерностью 2
def plot_lr_pca(X, y):
    model, X_pca, y = build_pca2_model(X,y)

    plt.scatter(X_pca[:,0],
                y,
                label='Fact cnt',
                c='green')

    plt.scatter(X_pca[:,0],
                model.predict(X_pca),
                label='Predicted cnt (PCA 2d)',
                c='red')

    plt.xlabel('Features')
    plt.ylabel('Cnt')
    plt.legend()

    return plt

plot_lr_pca(X, y).show()