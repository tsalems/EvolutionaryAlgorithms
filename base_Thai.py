"""This is the base code for all from ThaiNguyen
Copyright (c) 2020.
"""
import json
from configparser import ConfigParser
from os.path import isfile

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state, Bunch


def getConfig(section, option):
    # Read config.ini file
    config_object = ConfigParser()
    config_object.read("config.ini")
    # Get the info config
    # dataset = config_object["DATA-INPUT"]["dataset"]
    option_values = config_object.get(section, option)
    values = json.loads(option_values)
    return values


def avg(l):
    """
    Returns the average between list elements
    """
    return sum(l) / float(len(l))


def fetch_datasets(
        dataset_name=None,
        random_state=None,
        shuffle=True,
):
    filepath = "datasets\\" + dataset_name + ".txt"
    available = isfile(filepath)

    df = pd.read_table(filepath, header=None, sep=" ")
    # df.to_numpy()

    # encode labels column to numbers
    le = LabelEncoder()
    le.fit(df.iloc[:, -1])
    y = le.transform(df.iloc[:, -1])  # label
    X = df.iloc[:, :-1].to_numpy()  # data
    # X = df.iloc[:, :-1]  # data

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


def AccKFold(individual, X, y):
    print("_Accuracy Model with KFold_")

    if individual.count(0) != len(individual):
        # get index with value 0
        cols = [index for index in range(
            len(individual)) if individual[index] == 0]

        # X_df = pd.DataFrame(data=X[0:, 0:], )  # values

        # get features subset
        X_parsed = X.drop(X.columns[cols], axis=1)
        X_subset = pd.get_dummies(X_parsed).to_numpy()

        # KFold Cross Validation approach
        kf = KFold(n_splits=10, shuffle=True)
        # kf.split(X_subset)

        # Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to
        # this list
        accuracy_model_train = []
        accuracy_model_test = []

        # Iterate over each train-test split
        k = 1
        for train_index, test_index in kf.split(X_subset):
            # Split train-test
            X_train, X_test = X_subset[train_index], X_subset[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # Classifier
            clf = DecisionTreeClassifier()
            # Train the model
            model = clf.fit(X_train, y_train)
            # Append to accuracy_model the accuracy of the model
            acc_train = accuracy_score(y_train, model.predict(X_train), normalize=True)
            accuracy_model_train.append(acc_train)

            acc_test = accuracy_score(y_test, model.predict(X_test), normalize=True)
            accuracy_model_test.append(acc_test)

            print("k=" + str(k), "\t", acc_train, acc_test)
            k += 1

        # Print the accuracy
        accuracy_train = min(accuracy_model_train), avg(accuracy_model_train), max(accuracy_model_train)
        accuracy_test = min(accuracy_model_test), avg(accuracy_model_test), max(accuracy_model_test)
        # print(accuracy_model)

        # print("TRAIN: " + str(accuracy_train))
        # print("TEST: " + str(accuracy_test) + "\n")
        print("", "min", "avg", "max", sep="\t\t")
        print("TRAIN", accuracy_train[0], accuracy_train[1], accuracy_train[2], sep="\t")
        print("TEST", accuracy_test[0], accuracy_test[1], accuracy_test[2], sep="\t")

    return accuracy_train, accuracy_test
