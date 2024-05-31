import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.decomposition import PCA
import numpy as np

# Импорт файла и чтение данных
file = pd.read_csv('files/bikes_rent.csv')
X = file[['yr', 'season', 'temp']]
y = file[['cnt']]


def predict(year, season, temp):
    global X, y
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    model = LinearRegression()
    model.fit(X_pca, y)

    X_test = np.array([year, season, temp]).reshape(1, -1)
    X_test_pca = pca.transform(X_test)

    return model.predict(X_test_pca)


year = int(input('Введите год: '))
season = int(input('Введите сезон: '))
temp = int(input('Введите температуру: '))

print(f"Предсказанное значение: {predict(year, season, temp)}")
