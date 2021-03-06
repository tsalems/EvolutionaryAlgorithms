import logging
import random
import sys
from os.path import isfile

import numpy as np
import pandas as pd
from deap import creator, base, tools, algorithms
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier, Perceptron, \
    RidgeClassifier, RidgeClassifierCV, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.utils import check_random_state, Bunch

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S',
    level=logging.DEBUG,
    filename='AllClassificationML-log.txt'
)

LIST_CLASSIFIER = {
    "AdaBoostClassifier": AdaBoostClassifier,
    "BaggingClassifier": BaggingClassifier,
    "BernoulliNB": BernoulliNB,
    "CalibratedClassifierCV": CalibratedClassifierCV,
    "CategoricalNB": CategoricalNB,
    "ClassifierChain": ClassifierChain,
    "ComplementNB": ComplementNB,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "DummyClassifier": DummyClassifier,
    "ExtraTreeClassifier": ExtraTreeClassifier,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "GaussianNB": GaussianNB,
    "GaussianProcessClassifier": GaussianProcessClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    # "HistGradientBoostingClassifier":	HistGradientBoostingClassifier,
    "KNeighborsClassifier": KNeighborsClassifier,
    "LabelPropagation": LabelPropagation,
    "LabelSpreading": LabelSpreading,
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
    "LinearSVC": LinearSVC,
    "LogisticRegression": LogisticRegression,
    "LogisticRegressionCV": LogisticRegressionCV,
    "MLPClassifier": MLPClassifier,
    "MultiOutputClassifier": MultiOutputClassifier,
    "MultinomialNB": MultinomialNB,
    "NearestCentroid": NearestCentroid,
    "NuSVC": NuSVC,
    "OneVsOneClassifier": OneVsOneClassifier,
    "OneVsRestClassifier": OneVsRestClassifier,
    "OutputCodeClassifier": OutputCodeClassifier,
    "PassiveAggressiveClassifier": PassiveAggressiveClassifier,
    "Perceptron": Perceptron,
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,
    "RadiusNeighborsClassifier": RadiusNeighborsClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "RidgeClassifier": RidgeClassifier,
    "RidgeClassifierCV": RidgeClassifierCV,
    "SGDClassifier": SGDClassifier,
    "SVC": SVC,
    "StackingClassifier": StackingClassifier,
    "VotingClassifier": VotingClassifier,

}

LIST_DATASET = [
    # "iris",
    # "nuclear"
    # "Landsat7neighbour",
    "landsatImg",
    "sat.all",
    "ulc",

]


def avg(l):
    """
    Returns the average between list elements
    """
    return sum(l) / float(len(l))


def fetch_datasets(
        data_home=None,
        filter_data=None,
        download_if_missing=True,
        name_dataset=None,
        random_state=None,
        shuffle=False,
        verbose=False,

):
    # filename = "datasets\\Landsat7neighbour.txt"
    filename = "datasets\\" + name_dataset + ".txt"
    available = isfile(filename)

    df = pd.read_table(filename, header=None, sep=" ")
    # df = pd.read_table(filename, header=None, sep="\t")
    # df = pd.read_table(filename, header=None)
    # df = pd.read_table("datasets\\sat.all.txt", header=None, sep=" ")
    # df.to_numpy()

    # encode labels column to numbers
    le = LabelEncoder()
    le.fit(df.iloc[:, -1])
    y = le.transform(df.iloc[:, -1])  # label
    X = df.iloc[:, :-1].to_numpy()  # data

    if shuffle:
        ind = np.arange(X.shape[0])
        rng = check_random_state(random_state)
        rng.shuffle(ind)
        X = X[ind]
        y = y[ind]

    dataset = Bunch(data=X, target=y)

    return dataset


if __name__ == '__main__':

    logging.info('Started')

    i = 1
    for dataset in LIST_DATASET:

        logging.info("-" * 30 + str(i) + ".dataset:" + dataset + "-" * 30)
        print("-" * 30 + str(i) + ".dataset:" + dataset + "-" * 30)

        j = 1
        for classifier in LIST_CLASSIFIER:

            logging.info(str(j) + '.' + classifier)
            print(str(j) + '.' + classifier)

            satimage = fetch_datasets(name_dataset=dataset)
            X, y = satimage.data, satimage.target
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

            # kf = KFold(n_splits=10, random_state=None, shuffle=False)
            # kf.get_n_splits(X, y)
            # TEN TRAIN TEST

            clf = LIST_CLASSIFIER[classifier]()

            # model = clf.fit(X_train, y_train)
            # # scores = cross_val_score(clf, X, y, cv=10)
            #
            # y_pred = clf.predict(X_test)
            # # Calculate Accuracy
            # performance = accuracy_score(y_test, y_pred)
            #
            # logging.info("-" + classifier + ' performance: ' + str(performance))
            # print("-" + classifier + ' performance: ' + str(performance))

            # y_p = clf.predict(X_train)
            # # Calculate Accuracy
            # print(accuracy_score(y_train, y_p))

            # confusion matrix
            # cm_tree = confusion_matrix(y_test, y_pred_tree)
            # print(cm_tree)

            # print(scores)
            # print(recall_score(y_test, y_pred_tree))
            # print(precision_score(y_test, y_pred_tree))

            # -------------------------------------------------------------------------------------------------------
            logging.info("Accuracy Model with 10 KFold:")
            print("Accuracy Model with 10 KFold:")
            # KFold Cross Validation approach
            kf = KFold(n_splits=10, shuffle=True)
            kf.split(X)

            # Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to
            # this list
            accuracy_model_train = []
            accuracy_model_test = []

            # Iterate over each train-test split
            k = 1
            for train_index, test_index in kf.split(X):
                # Split train-test
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # Train the model
                model = clf.fit(X_train, y_train)
                # Append to accuracy_model the accuracy of the model
                acc_train = accuracy_score(y_train, model.predict(X_train), normalize=True)
                accuracy_model_train.append(acc_train)

                acc_test = accuracy_score(y_test, model.predict(X_test), normalize=True)
                accuracy_model_test.append(acc_test)

                print("k="+str(k), "\t", acc_train, acc_test)
                k += 1

            # Print the accuracy
            accuracy_train = avg(accuracy_model_train)
            accuracy_test = avg(accuracy_model_test)
            # print(accuracy_model)
            logging.info("TRAIN: " + str(accuracy_train))
            print("TRAIN: " + str(accuracy_train))
            logging.info("TEST: " + str(accuracy_test) + "\n")
            print("TEST: " + str(avg(accuracy_model_test)) + "\n")

            j += 1
        i += 1
        print("-" * 80)

    logging.info('Finished')
