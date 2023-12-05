# Competitie Kaggle.
# Link-uri: https://www.kaggle.com/competitions/playground-series-s3e16/overview
# https://www.kaggle.com/competitions/playground-series-s3e16/discussion/414157
# https://www.kaggle.com/competitions/playground-series-s3e16/discussion/413971
# https://www.kaggle.com/competitions/playground-series-s3e16/discussion/414138

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from seaborn import heatmap
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score


def remove_outliers(df):
    df['Height'] = df['Height'].replace(to_replace=0, value=np.nan)
    return df


def impute_nan(df):
    knn = KNNImputer(missing_values=np.nan)
    df['Height'] = knn.fit_transform(X=df['Height'].values.reshape(-1, 1))
    return df


def feature_engineering(df):
    df['Volume'] = df['Length'] * df['Diameter'] * df['Height']
    df['Density'] = df['Weight'] / df['Volume']
    df['Shell Percentage'] = df['Shell Weight'] / df['Weight']
    df['Meat Percentage'] = df['Shucked Weight'] / df['Weight']
    df['Viscera Percentage'] = df['Viscera Weight'] / df['Weight']
    df['Shell Surface Area'] = (df['Diameter'] * df['Length'] * 2) * (
            df['Diameter'] * df['Height'] * 2) * (
                                       df['Length'] * df['Height'] * 2)
    df['Shell Density'] = df['Shell Weight'] / df['Shell Surface Area']
    return df


train = pd.read_csv('Data\\train_extended.csv', index_col=0)
test = pd.read_csv('Data\\test.csv', index_col=0)
sub = pd.read_csv('Data\\sample_submission.csv', index_col=0)

print(train.head())
print(train.describe().to_string())
print(train.info())

train.hist()
plt.show()

train = remove_outliers(train)
test = remove_outliers(test)

train = impute_nan(train)
test = impute_nan(test)

train = pd.get_dummies(data=train, columns=['Sex'], drop_first=True)
test = pd.get_dummies(data=test, columns=['Sex'], drop_first=True)

train = feature_engineering(train)
test = feature_engineering(test)

heatmap(data=train.corr().round(2), annot=True)
plt.show()

X = train.drop(columns=['Age', 'Sex_F'])
y = train['Age']

param_grid = {'iterations': [2300, 2400]}
res = CatBoostRegressor(verbose=0, loss_function='MAE', depth=7).grid_search(param_grid=param_grid, X=X, y=y)

print(res['params'])

c = CatBoostRegressor(verbose=0, loss_function='MAE', depth=7, **res['params'])

cv = cross_val_score(estimator=c, X=X, y=y, scoring='neg_mean_absolute_error', cv=5)
print(-np.mean(cv))

c.fit(X=X, y=y)
y_pred = c.predict(data=test).round(0)

sub['Age'] = y_pred
sub.to_csv('Submission.csv')
