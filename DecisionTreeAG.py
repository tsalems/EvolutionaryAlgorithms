import random
import sys
from os.path import isfile

import numpy as np
import pandas as pd
from deap import creator, base, tools, algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state, Bunch


def avg(l):
    """
    Returns the average between list elements
    """
    return sum(l) / float(len(l))

def fetch_datasets(
    data_home=None,
    filter_data=None,
    download_if_missing=True,
    random_state=None,
    shuffle=True,
    verbose=False,

):

    filename = "datasets\\Landsat7neighbour.txt"
    available = isfile(filename)

    df = pd.read_table(filename, header=None, sep="\t")
    # df = pd.read_table("datasets\\sat.all.txt", header=None, sep=" ")
    # df.to_numpy()

    # encode labels column to numbers
    le = LabelEncoder()
    le.fit(df.iloc[:, -1])
    y = le.transform(df.iloc[:, -1])  # label
    X = df.iloc[:, :-1].to_numpy()  # data

    # data = np.load(filename)
    # print(data)
    # X, y = X["data"], y["label"]

    # numpy.set_printoptions(threshold=sys.maxsize)
    # print(X,y)
    # cnt=0
    # lst = data.files
    # for item in lst:
    #     cnt+=1
    #     print(item)
    #     print(data[item])
    #     print(cnt)

    if shuffle:
        ind = np.arange(X.shape[0])
        rng = check_random_state(random_state)
        rng.shuffle(ind)
        X = X[ind]
        y = y[ind]

    dataset = Bunch(data=X, target=y)

    return dataset

if __name__ == '__main__':

    # read dataframe from csv
    # df = pd.read_csv(dataframePath, sep=',')
    # df = pd.read_table("datasets\\ulc.data", header=None)
    # df = pd.read_table("datasets\\sat.all.txt", header=None, sep=" ")
    # df = pd.read_table("datasets\\Landsat7neighbour.txt", header=None, sep="\t")
    #
    # # encode labels column to numbers
    # le = LabelEncoder()
    # le.fit(df.iloc[:, -1])
    # y = le.transform(df.iloc[:, -1])  # label
    # X = df.iloc[:, :-1]  # data

    # # get features subset
    # X_parsed = X.drop(X.columns[cols], axis=1)
    # X_subset = pd.get_dummies(X_parsed)

    satimage = fetch_datasets()
    X, y = satimage.data, satimage.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

    # kf = KFold(n_splits=10, random_state=None, shuffle=False)
    # kf.get_n_splits(X, y)

    clf = DecisionTreeClassifier()

    clf.fit(X_train, y_train)
    # scores = cross_val_score(clf, X, y, cv=10)

    y_pred_tree = clf.predict(X_test)
    print('Decision tree classifier performance:')

    cm_tree = confusion_matrix(y_test, y_pred_tree)

    print(cm_tree)

    # Calculate Accuracy

    print(accuracy_score(y_test, y_pred_tree))

    # print(scores)

    # print(recall_score(y_test, y_pred_tree))

    # print(precision_score(y_test, y_pred_tree))

    # KFold Cross Validation approach
    kf = KFold(n_splits=10, shuffle=False)
    kf.split(X)

    # Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
    accuracy_model = []

    # Iterate over each train-test split
    for train_index, test_index in kf.split(X):
        # Split train-test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Train the model
        model = clf.fit(X_train, y_train)
        # Append to accuracy_model the accuracy of the model
        accuracy_model.append(accuracy_score(y_test, model.predict(X_test), normalize=True))

    # Print the accuracy
    print(accuracy_model)
    print(avg(accuracy_model))

