import pandas as pd
import numpy as np
import scipy as sp
import math
from string import join
import re

import json
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import *
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import *
from sklearn.linear_model import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
# from graphviz import Digraph
from sklearn import tree
import os
import requests

pd.options.mode.chained_assignment = None

# Loading the data as a DataFrame.
print 'Reading CSV'
# df_raw = pd.read_csv("~/rto-challenge-dataset.csv", encoding="ISO-8859-1")
df_raw = pd.read_table('~/rto-challenge-dataset.csv', sep=',', keep_default_na=True, na_values=['NA'])

print 'Reading CSV done!'
# print df_raw.drop_duplicates(subset=['feature_categorical_3'])

print 'df_raw.shape: ', df_raw.shape

# -----------------------Defining the column types--------------------
label_col = ['label']

feature_cols = ['feature_categorical_3', 'feature_categorical_4',
                'feature_categorical_5', 'feature_categorical_6', 'feature_categorical_7', 'feature_categorical_8',
                'feature_categorical_9', 'feature_categorical_10', 'feature_numerical_1', 'feature_numerical_2',
                'feature_numerical_3', 'feature_numerical_4', 'feature_numerical_5', 'feature_numerical_6',
                'feature_numerical_7', 'feature_numerical_8', 'feature_numerical_9', 'feature_numerical_10',
                'feature_numerical_11', 'feature_numerical_12', 'feature_numerical_13', 'feature_numerical_14',
                'feature_numerical_15', 'feature_numerical_16', 'feature_numerical_17', 'feature_numerical_18',
                'feature_numerical_19', 'feature_numerical_20', 'feature_numerical_21', 'feature_numerical_22',
                'feature_numerical_23', 'feature_numerical_24', 'feature_numerical_25', 'feature_numerical_26',
                'feature_numerical_27', 'feature_numerical_28', 'feature_numerical_29', 'feature_numerical_30',
                'feature_numerical_31', 'feature_numerical_32', 'feature_numerical_33', 'feature_numerical_34',
                'feature_numerical_35', 'feature_numerical_36', 'timestamp']

categorical_columns = ['feature_categorical_3',
                       'feature_categorical_4', 'feature_categorical_5', 'feature_categorical_6',
                       'feature_categorical_7', 'feature_categorical_8', 'feature_categorical_9',
                       'feature_categorical_10']

# ['feature_categorical_1', 'feature_categorical_2', 'feature_categorical_3',
#  'feature_categorical_4', 'feature_categorical_5', 'feature_categorical_6',
#  'feature_categorical_7', 'feature_categorical_8', 'feature_categorical_9',
#  'feature_categorical_10']

numerical_columns = ['feature_numerical_1', 'feature_numerical_2', 'feature_numerical_3', 'feature_numerical_4',
                     'feature_numerical_5', 'feature_numerical_6', 'feature_numerical_7', 'feature_numerical_8',
                     'feature_numerical_9', 'feature_numerical_10', 'feature_numerical_11', 'feature_numerical_12',
                     'feature_numerical_13', 'feature_numerical_14', 'feature_numerical_15', 'feature_numerical_16',
                     'feature_numerical_17', 'feature_numerical_18', 'feature_numerical_19', 'feature_numerical_20',
                     'feature_numerical_21', 'feature_numerical_22', 'feature_numerical_23', 'feature_numerical_24',
                     'feature_numerical_25', 'feature_numerical_26', 'feature_numerical_27', 'feature_numerical_28',
                     'feature_numerical_29', 'feature_numerical_30', 'feature_numerical_31', 'feature_numerical_32',
                     'feature_numerical_33', 'feature_numerical_34', 'feature_numerical_35', 'feature_numerical_36']

normal_numerical_feature_columns = ['feature_numerical_1', 'feature_numerical_2', 'feature_numerical_3',
                                    'feature_numerical_4', 'feature_numerical_5', 'feature_numerical_6',
                                    'feature_numerical_7', 'feature_numerical_8', 'feature_numerical_9',
                                    'feature_numerical_10', 'feature_numerical_11', 'feature_numerical_13',
                                    'feature_numerical_14', 'feature_numerical_15', 'feature_numerical_16',
                                    'feature_numerical_17', 'feature_numerical_18', 'feature_numerical_19',
                                    'feature_numerical_20', 'feature_numerical_21', 'feature_numerical_22',
                                    'feature_numerical_25', 'feature_numerical_26', 'feature_numerical_27',
                                    'feature_numerical_28', 'feature_numerical_29', 'feature_numerical_36']

mostly_nan_numerical_columns = ['feature_numerical_30', 'feature_numerical_31', 'feature_numerical_32',
                                'feature_numerical_33', 'feature_numerical_34']

categorically_disguised_numerical_columns = ['feature_numerical_12', 'feature_numerical_23', 'feature_numerical_24',
                                             'feature_numerical_35']

timestamp_column = ['timestamp']


# ------------------Numerical estimators-----------------------
# Transformer to impute missing values using mean of the column.
class MissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, col=None):
        self.mean = 0
        self.col = col

    def transform(self, X):
        X[self.col] = X[self.col].fillna(self.mean)
        return X

    def fit(self, X, y=None):
        self.mean = X[self.col].mean()
        return self


# Transformer that computes log(1+x) on the specified columns
class Log1p(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def transform(self, X):
        for col in self.cols: X[col] = X[col].apply(lambda x: math.log1p(x))
        return X

    def fit(self, X, y=None):
        return self


# Transfomer for performing one hot encoding on a specified categorical column
class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X):
        return pd.get_dummies(X)

    def fit(self, X, y=None):
        return self


class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# Transformer that bins the specified numeric column along the specified bins
class Binning(BaseEstimator, TransformerMixin):
    def __init__(self, col, bins):
        self.col = col
        self.bins = bins

    def transform(self, X):
        X[self.col] = [str(i) for i in np.digitize(X[self.col].values, self.bins)]
        return X

    def fit(self, X, y=None):
        return self


# ---------------------Selectors-----------------------------
# Stateless transformer for selecting a specified column
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col

    def transform(self, X):
        return X[self.col]

    def fit(self, X, y=None):
        return self


# Stateless transformer to selects one or more column from DataFrame.
class DFSubsetSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        return X[self.cols]

    def fit(self, X, y=None):
        return self


# --------------Converters-------------------------------
# Transformer to convert a DataFrame to a sparse matrix
class ConvertDFToMatrix(BaseEstimator, TransformerMixin):
    def transform(self, X):
        return sp.sparse.csr.csr_matrix(X.values)

    def fit(self, X, y=None):
        return self


# Transformer to convert a single column DataFrame to a sparse matrix
class ConvertDFToVector(BaseEstimator, TransformerMixin):
    def transform(self, X):
        return np.ravel(X.values)

    def fit(self, X, y=None):
        return self


class StringToTimeStampConverter(BaseEstimator, TransformerMixin):
    def transform(self, X):
        return pd.to_datetime(X)

    def fit(self, X, y=None):
        return self


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    def transform(self, X):
        if not isinstance(X, pd.Series):
            raise TypeError('The argument passed to TemporalFeatureExtractor must be a Pandas Series')
        if X.dtype.name != 'datetime64[ns]':
            raise TypeError(
                'The argument passed to TemporalFeatureExtractor must be a Pandas Series with dtype==datetime64[ns]')
        temporal_df_columns = ['month', 'day', 'dayofweek', 'hour', 'minute', 'weekofyear', 'quarter', 'dayofyear',
                               'epoch']
        _n_rows = X.shape[0]
        _n_columns = len(temporal_df_columns)
        _my_array = np.zeros((_n_rows, _n_columns))
        for idx, timestamp in enumerate(X):
            _my_array[idx, :] = [timestamp.month, timestamp.day, timestamp.dayofweek,
                                 timestamp.hour, timestamp.minute, timestamp.weekofyear,
                                 timestamp.quarter, timestamp.dayofyear, timestamp.value]
        # print 'Mean: ', _my_array[:, 8].mean()
        _my_array[:, 8] = _my_array[:, 8] - 1.47664006845e+18

        return pd.DataFrame(data=_my_array, columns=temporal_df_columns)

    def fit(self, X, y=None):
        return self


class MyDictVectoriser(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        return X[self.cols].to_dict(orient='str')

    def fit(self, X, y=None):
        return self


class FillNa(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.copy().fillna(0)


# ----------------------The model class ---------------------------------------------
class Model:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def train(self, training_data_X, training_data_Y):
        # self.pipeline = final_pipeline
        self.pipeline.fit(training_data_X, training_data_Y)
        # _tree = self.pipeline.steps[4][1].tree_
        #
        # from StringIO import StringIO
        # out = StringIO()
        # myTree = self.pipeline.steps[4][1].tree_
        # out = tree.export_graphviz(self.pipeline.steps[4][1], out_file='./graph.gv')
        # # tree.graph_from_dot_data(out.getvalue()).write_pdf("somefile.pdf")
        # # dot = Digraph(comment='The Round Table')
        # # dot.render('./graph.gv', view=True)
        # i=0

    def predict(self, input_json):
        record_df = pd.read_json(input_json)
        return json.dumps({'prediction': self.pipeline.predict_proba(record_df).tolist()[0][1]})

    def predict_local(self, input_df):
        prediction = self.pipeline.predict(input_df)
        return prediction


# --------------------Splitting the data-----------------
X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = train_test_split(df_raw[feature_cols], df_raw[label_col],
                                                                    test_size=.2, random_state=42, train_size=.8,
                                                                    stratify=df_raw[label_col])

print 'X_train_raw.shape: ', X_train_raw.shape
print 'X_test_raw.shape: ', X_test_raw.shape

print 'Y_train_raw.shape: ', Y_train_raw.shape
print 'Y_test_raw.shape: ', Y_test_raw.shape

df_train = X_train_raw.merge(Y_train_raw, left_index=True, right_index=True)
df_test = X_test_raw.merge(Y_test_raw, left_index=True, right_index=True)

# splitting training data into input and target
# obtaining the input features
df_train_X = df_train[feature_cols]
# obtaining the target
df_train_Y = df_train[label_col]

# creating a dict object to store the true labels for train and test cases
# actual_Y={}
# transforming the target
actual_Y = {'train': df_train_Y[label_col]}

df_test_X = df_test[feature_cols]

# obtaining target and transforming it
df_test_Y = df_test[label_col]

# transforming the test target as ell
actual_Y['test'] = df_test_Y[label_col]

# --------------------------------- Pipeline and train --------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import *
from sklearn.ensemble import *
from sklearn.model_selection import GridSearchCV

# learner = AdaBoostRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
# learner = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=1, random_state=0)
# learner = TheilSenRegressor()
# learner = BernoulliNB()
clf1 = LogisticRegression()
clf2 = RandomForestClassifier(n_jobs=-1, n_estimators=100, max_depth=13, random_state=42, min_samples_split=6,
                              min_impurity_split=0.05, bootstrap=False)
clf3 = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=2, random_state=0)
# clf4 = AdaBoostRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
# clf5 = KNeighborsClassifier()

learner = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3)],
                           voting='soft', n_jobs=-1,
                           weights=[1, 3, 1])

learner2 = DecisionTreeClassifier(min_samples_split=10)
learner3 = MLPClassifier(hidden_layer_sizes=(100, 50, 10), verbose=10)

classifiers = [clf1, clf2, clf3, learner, learner2, learner3]
for clf in classifiers:
    # --------------------Constructing the pipelines-----------
    numerical_feature_extractor = Pipeline([
        ('selector', DFSubsetSelector(normal_numerical_feature_columns)),
        ('imputer', Imputer())
        # ('minmax_scaler', MinMaxScaler()) # Note: MinMax normalization was actually reducing the ROC by 0.3 percent.
        # ('df_to_matrix', ConvertDFToMatrix())
    ])

    mostly_nan_numerical_feature_extractor = Pipeline([
        ('selector', DFSubsetSelector(mostly_nan_numerical_columns)),
        ('fillna', FillNa())  # instead of imputing mean, we impute zeros.
    ])

    categorical_feature_extractor = Pipeline([
        ('selector', DFSubsetSelector(categorical_columns)),
        # ('dict_vectorizer', DictVectorizer())
        # ('one_hot_encoder', OneHotEncoder())
        # ('imputer', Imputer(strategy='median')),
        ('label_encoder', MultiColumnLabelEncoder(categorical_columns))
    ])

    timestamp_feature_extractor = Pipeline([
        ('selector', ColumnSelector('timestamp')),
        ('string_timestamp_converter', StringToTimeStampConverter()),
        ('temporal_feature_extractor', TemporalFeatureExtractor())
    ])

    all_feature_extractor_preprocessor = FeatureUnion(
        transformer_list=[('numerical_column_extractor', numerical_feature_extractor),
                          ('categorical_column_extractor', categorical_feature_extractor),
                          ('temporal_feature_extractor', timestamp_feature_extractor),
                          ('mostly_nan_feature_extractor', mostly_nan_numerical_feature_extractor)
                          ])

    # feature_selector = SelectKBest(score_func=chi2, k=10)
    feature_selector = SelectPercentile(score_func=f_classif, percentile=45)
    # feature_selector = SelectFromModel(
    #     RandomForestClassifier(n_jobs=-1, n_estimators=100, max_depth=13, random_state=42, min_samples_split=6,
    #                            min_impurity_split=0.05, bootstrap=False))
    # feature_selector = SelectFromModel(estimator=ExtraTreesClassifier())

    final_pipeline = Pipeline([('feature_extractor_preprocessor', all_feature_extractor_preprocessor),
                               ('imputer', Imputer()),
                               ('feature_selector', feature_selector),
                               ('polynomial', PolynomialFeatures(degree=2)),
                               ('learner', clf)])

    # ---------------------------------------------Triggering the training------------------------------
    print '------------------------------------'
    print 'Results for classifier: ', type(clf)
    model = Model(final_pipeline)
    print 'Training...'
    # print 'Sample training rows: ', df_train_X.head(3)
    # print 'Training data shape: ', df_train_X.shape
    model.train(df_train_X, actual_Y['train'])
    print 'Training done.'

    print 'Predicting a test record...'
    test_record = '[{"timestamp": "2016-10-13 17:00:00","feature_numerical_20": 0.798,"feature_numerical_22": 0.74,"feature_categorical_5": "9ae17c6551b342d5d71f080e0099fae46f861342","feature_numerical_21": 0.00132974832762,"feature_categorical_6": "b87025b357ed093c17f9999cefcf4baa93d02a70","feature_numerical_25": 0.0483253588517,"feature_numerical_30": 0.040404040404,"feature_numerical_6": 0,"feature_numerical_24": 0.989629964876,"feature_numerical_31": 0.0839694656489,"feature_numerical_34": 0.0366972477064,"feature_numerical_2": 0.0357142857143,"feature_categorical_1": "0822e3e95f846d2b81629d537615d9101db7d0c5","feature_categorical_3": "e4c15bf37310ad233fee194de3f00fbd2f91dee1","feature_numerical_36": 0.00995836802664,"feature_categorical_7": "e5353879bd69bfddcb465dad176ff52db8319d6f","feature_numerical_26": 0.00096468348751,"feature_categorical_9": "b85ab32eaa572c8016edf68011078dceed8149e5","feature_categorical_2": "d832ad51b52348d11415d900454cb72d944162ff","feature_numerical_28": 0.219459459459,"feature_numerical_12": 0,"feature_numerical_29": 0.0670731707317,"feature_numerical_16": 0.225806451613,"feature_categorical_4": "88b33e4e12f75ac8bf792aebde41f1a090f3a612","feature_numerical_3": 0.00361010830325,"feature_categorical_8": "89f1ebf5ace10fe4d43c85a7ad419905164b9883","feature_numerical_8": 0.0881979695431,"feature_numerical_19": 0.001,"feature_numerical_10": 0.001,"feature_numerical_4": 0.3337,"feature_numerical_1": 0,"feature_numerical_35": 0.166666666667,"feature_numerical_32": 0.00676818950931,"feature_numerical_23": 0.989689806228,"feature_numerical_11": 0,"feature_numerical_5": 0.0033,"feature_numerical_27": 0,"feature_numerical_13": 0.0165796360247,"feature_numerical_15": 0.0234908389585,"feature_numerical_33": 0,"feature_categorical_10": "d166e844a3f3f87149cc4f866eb998e9a751c72a","feature_numerical_9": 0.483544107247,"feature_numerical_14": 0.0148434759981,"feature_numerical_18": 0.00407763823194,"feature_numerical_7": 0.0192923007155,"feature_numerical_17": 0.00598011960239}]'
    print model.predict(test_record)
    print 'Predicting a test record... Done'

    print 'Predicting the test set...'
    print 'Pipeline Score: ', final_pipeline.score(df_test_X, actual_Y['test'])

    pred_prob_Y = {}
    pred_prob_Y['test'] = final_pipeline.predict_proba(df_test_X)[:, 1]
    print 'ROC_AUC Score: ', metrics.roc_auc_score(actual_Y['test'], pred_prob_Y['test'])

# ------------------------------------Publishing---------------------------------------------

# set environment variable
os.environ["ML_SDK_CONF_BUCKET"] = "ml-challenge-sdk"

# print 'Publishing to ModelHost...'
# from mlsdk.MLApi import MLApi
#
# print MLApi().modelhost_publish_new_model(model, "yaML")
# print 'Published.'
