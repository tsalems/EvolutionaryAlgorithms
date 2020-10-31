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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
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

LIST_CLASSIFIER = [
    "AdaBoostClassifier",
    "BaggingClassifier",
    "BernoulliNB",
    "CalibratedClassifierCV",
    "CategoricalNB",
    "ClassifierChain",
    "ComplementNB",
    "DecisionTreeClassifier",
    "DummyClassifier",
    "ExtraTreeClassifier",
    "ExtraTreesClassifier",
    "GaussianNB",
    "GaussianProcessClassifier",
    "GradientBoostingClassifier",
    #"HistGradientBoostingClassifier",
    "KNeighborsClassifier",
    "LabelPropagation",
    "LabelSpreading",
    "LinearDiscriminantAnalysis",
    "LinearSVC",
    "LogisticRegression",
    "LogisticRegressionCV",
    "MLPClassifier",
    "MultiOutputClassifier",
    "MultinomialNB",
    "NearestCentroid",
    "NuSVC",
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OutputCodeClassifier",
    "PassiveAggressiveClassifier",
    "Perceptron",
    "QuadraticDiscriminantAnalysis",
    "RadiusNeighborsClassifier",
    "RandomForestClassifier",
    "RidgeClassifier",
    "RidgeClassifierCV",
    "SGDClassifier",
    "SVC",
    "StackingClassifier",
    "VotingClassifier",

]

LIST_DATASET = [
    "iris",
    "nuclear"
    "Landsat7neighbour",
    "landsatImg"
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
        random_state=None,
        shuffle=True,
        verbose=False,

):
    filename = "datasets\\Landsat7neighbour.txt"
    available = isfile(filename)

    df = pd.read_table(filename, header=None, sep="\t")
    # df = pd.read_table("datasets\\sat.all.txt", header=None, sep=" ")
    df = pd.read_table("datasets\\ulc.data", header=None)
    # df.to_numpy()

    # encode labels column to numbers
    le = LabelEncoder()
    le.fit(df.iloc[:, -1])
    y = le.transform(df.iloc[:, -1])  # label
    X = df.iloc[:, :-1].to_numpy()  # data
    # X = df.iloc[:, :-1]  # data

    if shuffle:
        ind = np.arange(X.shape[0])
        rng = check_random_state(random_state)
        rng.shuffle(ind)
        X = X[ind]
        y = y[ind]

    dataset = Bunch(data=X, target=y)

    return dataset


def getFitness(individual, X, y):
    """
    Feature subset fitness function
    """

    if individual.count(0) != len(individual):
        # get index with value 0
        cols = [index for index in range(
            len(individual)) if individual[index] == 0]

        # get features subset
        X_parsed = X.drop(X.columns[cols], axis=1)
        X_subset = pd.get_dummies(X_parsed)

        # X_subset = X
        #
        # for col in cols:
        #     X_subset[col].values[:] = 0

        # apply classification algorithm
        clf = AdaBoostClassifier()
        clf = BaggingClassifier()
        clf = BernoulliNB()

        clf = CalibratedClassifierCV()
        clf = CategoricalNB()
        clf = ClassifierChain()
        clf = ComplementNB()

        clf = DecisionTreeClassifier()
        clf = DummyClassifier()

        clf = ExtraTreeClassifier()
        clf = ExtraTreesClassifier()

        clf = GaussianNB()
        clf = GaussianProcessClassifier()
        clf = GradientBoostingClassifier()

        # clf = HistGradientBoostingClassifier()

        clf = KNeighborsClassifier()

        clf = LabelPropagation()
        clf = LabelSpreading()
        clf = LinearDiscriminantAnalysis()
        clf = LinearSVC()
        clf = LogisticRegression()
        clf = LogisticRegressionCV()

        clf = MLPClassifier()
        clf = MultiOutputClassifier()
        clf = MultinomialNB()

        clf = NearestCentroid()
        clf = NuSVC()

        clf = OneVsOneClassifier()
        clf = OneVsRestClassifier()
        clf = OutputCodeClassifier()

        clf = PassiveAggressiveClassifier()
        clf = Perceptron()

        clf = QuadraticDiscriminantAnalysis()

        clf = RadiusNeighborsClassifier()
        clf = RandomForestClassifier()
        clf = RidgeClassifier()
        clf = RidgeClassifierCV()

        clf = SGDClassifier()
        clf = SVC()
        clf = StackingClassifier()

        clf = VotingClassifier()

        # clf.fit(X, y)
        # clf.fit(X_subset, y_train)
        clf.fit(X_subset, y)

        # y_pred_ANN = clf.predict(X_test)
        # y_pred = clf.predict(X_subset)

        # score = cross_val_score(clf, X, y, cv=5)
        #
        # print(max(score), min(score))

        return (avg(cross_val_score(clf, X_subset, y, cv=5)),)
        # return (avg(score),)
        # return accuracy_score(y, y_pred_ANN)
    else:
        return (0,)


def geneticAlgorithm(X, y, n_population, n_generation):
    """
    Deap global variables
    Initialize variables to use eaSimple
    """
    # create individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # create toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, len(X.columns))
    # toolbox.register("individual", tools.initRepeat,
    #                  creator.Individual, toolbox.attr_bool, X.shape[1])
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)
    toolbox.register("evaluate", getFitness, X=X, y=y)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # initialize parameters
    pop = toolbox.population(n=n_population)
    hof = tools.HallOfFame(n_population * n_generation)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # genetic algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.9, mutpb=0.05,
                                   ngen=n_generation, stats=stats, halloffame=hof,
                                   verbose=True)

    # return hall of fame
    return hof


def bestIndividual(hof, X, y):
    """
    Get the best individual
    """
    maxAccurcy = 0.0
    for individual in hof:
        # if (individual.fitness.values > maxAccurcy):
        if individual.fitness.values[0] > maxAccurcy:
            maxAccurcy = individual.fitness.values
            _individual = individual

    _individualHeader = [list(X)[i] for i in range(
        len(_individual)) if _individual[i] == 1]
    return _individual.fitness.values, _individual, _individualHeader


if __name__ == '__main__':
    n_pop = 50
    n_gen = 50

    satimage = fetch_datasets()
    X, y = satimage.data, satimage.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

    X_train_df = pd.DataFrame(data=X_train[0:, 0:], )  # values
    # index = X_train[1:, 0],  # 1st column as index
    # columns = X_train[0, 1:])  # 1st row as the column names

    X_test_df = pd.DataFrame(data=X_test[0:, 0:], )  # values

    X_df = pd.DataFrame(data=X[0:, 0:], )

    # get accuracy with all features
    # individual = [1 for i in range(X.shape[1])]
    individual = [1 for i in range(len(X_df.columns))]
    print("Train with all features: \t" +
          str(getFitness(individual, X_train_df, y_train)) + "\n")

    print("Test with all features: \t" +
          str(getFitness(individual, X_test_df, y_test)) + "\n")

    # apply genetic algorithm
    hof = geneticAlgorithm(X_train_df, y_train, n_pop, n_gen)

    # select the best individual
    accuracy, individual, header = bestIndividual(hof, X, y)
    print('Best Accuracy: \t' + str(accuracy))
    print('Number of Features in Subset: \t' + str(individual.count(1)))
    print('Individual: \t\t' + str(individual))
    print('Feature Subset\t: ' + str(header))

    print("Test with subset features: \t" +
          str(getFitness(individual, X_test_df, y_test)) + "\n")
