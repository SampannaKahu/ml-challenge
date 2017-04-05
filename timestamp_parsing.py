'''
This file is basically a scratch-pad used to quickly try-out new code before productionising it.
'''

import dateutil.parser
from datetime import date
# from Pandas import Series
import pandas as pd
import numpy as np
from pandas.tslib import Timestamp
import json

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder

dd = dateutil.parser.parse('2016-10-13 17:00:00')

ordinal = dd.toordinal()

d = date.fromordinal(ordinal)

print 'Weekday: ', d.weekday()
print 'Date: ', d.day
print 'Month: ', d.month
s = pd.to_datetime(pd.Series(['2016-10-13 17:00:00', '2017-11-14 18:00:00', None]))
print s

# Loading the data as a DataFrame.
print 'Reading CSV'
# df_raw = pd.read_csv("~/rto-challenge-dataset.csv", encoding="ISO-8859-1")
df_raw = pd.read_table('~/rto-challenge-dataset.csv', sep=',', keep_default_na=True, na_values=['NA'], parse_dates=True,
                       index_col='timestamp', )
# df_with_new_index = df_raw.set_index('timestamp')
# abc = df_raw['timestamp']
# fgh = pd.to_datetime(abc)
i = 0

classes, y = np.unique(df_raw['feature_categorical_5'], return_inverse=True)

# a = np.array([[1, 3, 4], [1, 2, 3], [1, 2, 1]])
# print a
# a[1, :] = [0, 0, 0]
# print a

# _my_array = np.zeros((fgh[0:10].shape[0], 8))
# for idx, timestamp in enumerate(fgh[0:10]):
#     _my_array[idx, :] = [timestamp.month, timestamp.day, timestamp.dayofweek,
#                          timestamp.hour, timestamp.minute, timestamp.weekofyear, timestamp.quarter, timestamp.dayofyear]

print json.dumps({'prediction': 0.1234})

print np.linspace(start=0.01, stop=0.1, num=10, endpoint=True)

categorical_columns = ['feature_categorical_1', 'feature_categorical_2', 'feature_categorical_3',
                       'feature_categorical_4', 'feature_categorical_5', 'feature_categorical_6',
                       'feature_categorical_7', 'feature_categorical_8', 'feature_categorical_9',
                       'feature_categorical_10']

numerical_columns = ['feature_numerical_1', 'feature_numerical_2', 'feature_numerical_3', 'feature_numerical_4',
                     'feature_numerical_5', 'feature_numerical_6', 'feature_numerical_7', 'feature_numerical_8',
                     'feature_numerical_9', 'feature_numerical_10', 'feature_numerical_11', 'feature_numerical_12',
                     'feature_numerical_13', 'feature_numerical_14', 'feature_numerical_15', 'feature_numerical_16',
                     'feature_numerical_17', 'feature_numerical_18', 'feature_numerical_19', 'feature_numerical_20',
                     'feature_numerical_21', 'feature_numerical_22', 'feature_numerical_23', 'feature_numerical_24',
                     'feature_numerical_25', 'feature_numerical_26', 'feature_numerical_27', 'feature_numerical_28',
                     'feature_numerical_29', 'feature_numerical_30', 'feature_numerical_31', 'feature_numerical_32',
                     'feature_numerical_33', 'feature_numerical_34', 'feature_numerical_35', 'feature_numerical_36']

categorically_disguised_numerical_columns = ['feature_numerical_12', 'feature_numerical_23', 'feature_numerical_24',
                                             'feature_numerical_35']


class MyCategorizationClass(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        if not type(X) is pd.DataFrame:
            TypeError('Only a Pandas DataFrame can be passed to MyCategorizationFunction.transform()')
        X = pd.DataFrame(X)
        for col in self.cols:
            s = pd.Series(X[col], dtype="category")
            # print 'Feature name: ', col, ',  number of unique categories: ', len(s.cat.categories), ' out of total ', \
            #     s.shape[0], ' samples.'




            # self.categories = s.cat.categories
            # print col, s.values.unique
            # s.cat.categories = [self.myFunct(g) for g in s.cat.categories]
            pass

    def fit(self, X, y=None):
        return self


class MyNumericalTestClass(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        if not type(X) is pd.DataFrame:
            TypeError('Only a Pandas DataFrame can be passed to MyCategorizationFunction.transform()')
        X = pd.DataFrame(X)
        for col in self.cols:
            s = pd.Series(X[col])
            # s.plot(kind='hist')
            s.plot()
            # print 'Feature name: ', col, ',  number of unique categories: ', len(s.cat.categories), ' out of total ', \
            #     s.shape[0], ' samples.'




            # self.categories = s.cat.categories
            # print col, s.values.unique
            # s.cat.categories = [self.myFunct(g) for g in s.cat.categories]
            pass

    def fit(self, X, y=None):
        return self


# Transformer that computes log(1+x) on the specified columns
class Log1p(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def transform(self, X):
        for col in self.cols: X[col] = X[col].apply(lambda x: np.math.log1p(x))
        return X

    def fit(self, X, y=None):
        return self


# print df_raw.isnull().sum()


# df_raw[numerical_columns].plot()
# clf = MyNumericalTestClass(categorically_disguised_numerical_columns)
# clf = MyCategorizationClass(categorical_columns)
# clf.transform(df_raw)

# clf.transform(df_raw)

# df = pd.DataFrame(
#     {'cities': ['mumbai', 'delhi', 'calcutta', 'calcutta'], 'weather': ['sunnny', 'rainy', 'cloudy', 'cloudy']})
#
# one_hot_cities = pd.get_dummies(df['cities'])
# output = pd.DataFrame()
# output.append(pd.get_dummies(df['cities']))
# output.append(pd.get_dummies(df['weather']))
# output.append(pd.get_dummies(df))
# dummies_df = pd.get_dummies(df)
# one_hot_columns = pd.get_dummies(df).columns
# cols_to_retain = ['cities_mumbai', 'weather_cloudy']

# indices_to_retain = [one_hot_columns.tolist().index(col) for col in cols_to_retain]
# retained_dummies = pd.get_dummies(df).values[:, indices_to_retain]

fit_df = pd.DataFrame(
    {'cities': ['mumbai', 'delhi', 'calcutta', 'calcutta'], 'weather': ['sunnny', 'rainy', 'cloudy', 'cloudy']})
transform_df = pd.DataFrame(
    {'cities': ['calcutta', 'delhi', 'calcutta', 'calcutta', 'delhi'],
     'weather': ['sunnny', 'rainy', 'cloudy', 'cloudy', 'winter']})

fit_dummies = pd.get_dummies(fit_df)
fit_columns = fit_dummies.columns
transform_dummies = pd.get_dummies(transform_df)
transform_columns = transform_dummies.columns
cols_to_drop = [col for col in transform_columns if col not in fit_columns]
cols_to_add = [col for col in fit_columns if col not in transform_columns]
transform_dummies = transform_dummies.drop(cols_to_drop, axis='columns')
df_to_add = pd.DataFrame(0, index=transform_dummies.index, columns=cols_to_add)
transform_dummies = transform_dummies.join(df_to_add)
n_features = 1
ret_val = transform_dummies.loc[transform_dummies.sum(axis=1) != n_features]
j = 0
#
# from sklearn.preprocessing import OneHotEncoder
#
#
# class MultiColumnLabelEncoder:
#     def __init__(self, columns=None):
#         self.columns = columns  # array of column names to encode
#
#     def fit(self, X, y=None):
#         return self  # not relevant here
#
#     def transform(self, X):
#         '''
#         Transforms columns of X specified in self.columns using
#         LabelEncoder(). If no columns specified, transforms all
#         columns in X.
#         '''
#         output = X.copy()
#         if self.columns is not None:
#             for col in self.columns:
#                 output[col] = LabelEncoder().fit_transform(output[col])
#         else:
#             for colname, col in output.iteritems():
#                 output[colname] = LabelEncoder().fit_transform(col)
#         return output
#
#     def fit_transform(self, X, y=None):
#         return self.fit(X, y).transform(X)
#
#
# columns = df.columns
# clf1 = MultiColumnLabelEncoder(columns=columns)
# intermediate_data = clf1.transform(df)
# clf2 = OneHotEncoder(dtype=str)
# intermediate_data = clf2.fit(intermediate_data)
#
# o = clf2.fit(df)
#
i = 0
