import pandas as pd
from sklearn.linear_model import Lasso
from sklearn. linear_model import LassoCV
from sklearn. model_selection import RepeatedKFold
from numpy import arange

# Импорт файла и чтение данных
file = pd.read_csv('files/bikes_rent.csv')
X = file[['yr', 'season', 'temp']]
y = file[['cnt']]

def search_most_infl_feature(X, y):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    cv_model = LassoCV(alphas=arange(0, 1, 0.01), cv=cv, n_jobs=-1)
    cv_model.fit(X,y)

    index = abs(cv_model.coef_).argmax()
    max_value = X.columns[index]

    return (index, max_value)

print(search_most_infl_feature(X, y))

