import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np

# Импорт файла и чтение данных
file = pd.read_csv('files/bikes_rent.csv')

season = file['season']
weathersit = file['weathersit']
temp = file['temp']
cnt = file['cnt']

# Вычисляем точность простой линейрой регрессии
def get_lr_simple_accuracy(X, y):
    model = LinearRegression()
    model.fit(X,y)

    # Вычисляем точность
    accuracy = model.score(X,y)
    return accuracy

# Вычисляем точность PCA 2d
def get_pca2_accuracy(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    model = LinearRegression()
    model.fit(X_pca,y)

    # Вычисляем точность
    accuracy = model.score(X_pca,y)
    return accuracy



# # Выводим диаграмму PCA с размерностью 2
# plot_lr_pca(
#     X=file[['yr','season', 'temp', 'windspeed(ms)']],
#     y=file[['cnt']]
# ).show()

# # Находим признак с наибольшим абсолютным коэффициентом
# most_infl_feature = search_most_infl_feature(
#     X=file.drop(columns=['cnt']),
#     y=file[['cnt']],
#     alpha=0.1
# )
# print(most_infl_feature)

# # Вычисляем точность модели простой линейной регрессии
# lr_simple_accuracy = get_lr_simple_accuracy(
#     X=file[['yr','season', 'temp', ]],
#     y=file[['cnt']]
# )
# print(f"lr_simple_accuracy: {lr_simple_accuracy}")

# Вычисляем точность модели PCA с размерностью пространства 2
lr_pca_accuracy = get_pca2_accuracy(
    X=file[['yr','season', 'temp',
            ]],
    y=file[['cnt']]
)

print(f"lr_pca_accuracy: {lr_pca_accuracy}")



