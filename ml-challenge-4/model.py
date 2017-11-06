#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', 1000)

train = pd.read_csv('data/train_data.csv')
test = pd.read_csv('data/test_data.csv')
sub = pd.read_csv('data/sample_submission.csv')

feature_names = [x for x in train.columns if x not in ['connection_id','target']]
target = train['target']

sample_leaf_options = ["auto"]
accuracies = []

def run_forest(leaf_size):  
    print("Samples Leaf: {}".format(leaf_size))
    clf = RandomForestClassifier(n_estimators = 5000, 
                                 oob_score = True, 
				 max_features = "auto",
                                 n_jobs = -1,
                                 random_state = 32,
				 min_samples_split = 2,
                                 min_samples_leaf = 6)

    trained_model = clf.fit(train[feature_names], target)

    #print("Trained model :: ", trained_model)

    predictions = trained_model.predict(test[feature_names])
    actual_targets = train.head(91166)['target'].astype(int).tolist()

    print("OOB Score :: {}".format(clf.oob_score_))
    print("Train Accuracy :: {}".format(accuracy_score(actual_targets, predictions)))
    accuracies.append(accuracy_score(actual_targets, predictions))

    #print(list(zip(train[feature_names], clf.feature_importances_)))
    
    cm = pd.DataFrame(confusion_matrix(actual_targets, predictions), columns=[0,1,2], index=[0,1,2])
    #print(cm)
    
    sub['target'] = predictions
    sub['target'] = sub['target'].astype(int)
    sub.to_csv('submissions/random_forest.csv', index=False)
    return
  
for leaf_size in sample_leaf_options:
    run_forest(leaf_size)

plt.plot(sample_leaf_options, accuracies, 'ro')
plt.show()
