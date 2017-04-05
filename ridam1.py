from sklearn.neural_network import MLPClassifier
import pandas as pd
# import sys
import math
import time
import datetime
import random
from sklearn import metrics
#
# if len(sys.argv) < 2:
#     print "please provide input dataset file path as argument."
#     sys.exit(0)

# initialise variables
feature_categorical_map = dict()

data_file = '~/rto-challenge-dataset.csv'
df = pd.read_csv(data_file)
headers = list(df)
for header in headers:
    if str(header).startswith("feature_c"):
        # if str(header) == "feature_categorical_1" or str(header) == "feature_categorical_2":
        #    continue
        value_set = set([value for value in df[header]])
        feature_categorical_map[header] = dict()
        for index, value in enumerate(value_set):
            feature_categorical_map[header][value] = float(index) / len(value_set)

X = []
Y = []
epoch = datetime.datetime.fromtimestamp(0)
ts = time.time()
print '\n\nstarting buidlding X and Y'
for row in df.iterrows():
    x = []
    for row_item in row[1].iteritems():
        column_name = str(row_item[0])
        feature_value = row_item[1]
        if column_name.startswith("feature_c"):
            # if column_name == "feature_categorical_1" or column_name == "feature_categorical_2":
            #    continue
            x.append(feature_categorical_map[column_name][feature_value])
        elif column_name.startswith("feature_n"):
            if math.isnan(feature_value):
                feature_value = 0.0
            x.append(feature_value)
        elif column_name.startswith('t'):
            dt = datetime.datetime.strptime(feature_value, "%Y-%m-%d %H:%M:%S")
            x.append((dt - epoch).total_seconds() / ts)
        elif column_name.startswith('l'):
            Y.append(feature_value)
    X.append(x)
random.shuffle(X)
index = len(X) / 10
X_test = X[0:index]
Y_test = Y[0:index]
X = X[index + 1:]
Y = Y[index + 1:]
print "\n\nStarting training model"
clf = MLPClassifier(learning_rate='adaptive',
                    solver='lbfgs',
                    max_iter=1000,
                    verbose=10,
                    hidden_layer_sizes=(200, 200)
                    )

clf.fit(X, Y)
pred_prob = clf.predict_proba(X_test)[:, 1]
print 'ROC_AUC Score: ', metrics.roc_auc_score(Y_test, pred_prob)

'''counter = 0

for index,y in enumerate(Y_predict):
    if y == Y_test[index]:
        counter += 1
print "accuracy: ", float(counter)/len(Y_test)
print "one_perc: ", (float(len([bit for bit in Y_predict if bit == 1]))*100)/len(Y_test)'''
