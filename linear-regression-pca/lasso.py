import pandas as pd
from sklearn.linear_model import Lasso

# Импорт файла и чтение данных
file = pd.read_csv('files/bikes_rent.csv')
X = file[['yr', 'season', 'temp']]
y = file[['cnt']]

def search_most_infl_feature(X, y, alpha):
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)

    index = abs(lasso.coef_).argmax()
    max_value = X.columns[index]

    return (index, max_value)

print(search_most_infl_feature(X, y, alpha=0.1))

