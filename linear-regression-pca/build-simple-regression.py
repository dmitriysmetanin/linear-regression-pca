import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# Импорт файла и чтение данных
file = pd.read_csv('files/bikes_rent.csv')
weathersit = file['weathersit']
cnt = file['cnt']


def plot_lr(weathersit, cnt):
    model = LinearRegression()
    model.fit(weathersit.values.reshape(-1, 1), cnt)
    print(f"Coefficients: {model.coef_}")

    predictions = model.predict(weathersit.values.reshape(-1, 1))

    plt.scatter(weathersit, cnt, c='blue')
    plt.plot(weathersit, predictions, color='red', label='Linear Regression')
    plt.xlabel('Weathersit')
    plt.ylabel('Cnt')
    plt.legend()

    return plt


weathersit_cnt_plot = plot_lr(weathersit, cnt)
weathersit_cnt_plot.show()
