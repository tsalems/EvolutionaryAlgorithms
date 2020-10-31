import random
import sys
from os.path import isfile

import numpy as np
import pandas as pd
from deap import creator, base, tools, algorithms
from keras.metrics import accuracy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
        filename=None,
        data_home=None,
        filter_data=None,
        download_if_missing=True,
        random_state=None,
        shuffle=True,
        verbose=False,

):
    # filename = "datasets\\Landsat7neighbour.txt"
    # filename = "datasets\\landsatImg.txt"
    # filename = "datasets\\sat.all.txt"
    # filename = "datasets\\ulc.txt"
    #
    # available = isfile(filename)

    df = pd.read_table(filename, header=None, sep=" ")
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


def getFitness(individual, X, y):
    """
    Feature subset fitness function
    """

    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

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
        # clf = LogisticRegression()
        # clf = MLPClassifier()
        # clf = MLPClassifier(hidden_layer_sizes=(6,),max_iter=200,learning_rate_init=0.02,momentum=0.01)
        # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        clf = DecisionTreeClassifier()

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
            maxAccurcy = individual.fitness.values[0]
            _individual = individual

    _individualHeader = [list(X)[i] for i in range(
        len(_individual)) if _individual[i] == 1]
    return _individual.fitness.values, _individual, _individualHeader


if __name__ == '__main__':

    # filename = "datasets\\Landsat7neighbour.txt"
    # filename = "datasets\\landsatImg.txt"
    filename = "datasets\\sat.all.txt"
    # filename = "datasets\\ulc.txt"

    available = isfile(filename)
    if available:
        f = open("GA-result.txt", "a")
    else:
        f = open("GA-result.txt", "w")

    f.write("\n" + "-" * 100)
    now = pd.datetime.datetime.now()
    f.write(now.strftime('\n%d/%m/%Y %H:%M:%S\n'))
    f.write("-" * 20 + "\n")

    logbook = tools.Logbook()
    logbook.header = ['Dataset', 'avgTrain', 'minFull', 'avgFull', 'maxFull', 'min', 'avg', 'max', 'AllFeature',
                      'SelectedFeature', 'BestIndividual']

    n_pop = 5
    n_gen = 5

    satimage = fetch_datasets(filename=filename)
    X, y = satimage.data, satimage.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

    # X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.1)

    X_train_df = pd.DataFrame(data=X_train[0:, 0:], )  # values
    # index = X_train[1:, 0],  # 1st column as index
    # columns = X_train[0, 1:])  # 1st row as the column names

    X_test_df = pd.DataFrame(data=X_test[0:, 0:], )  # values

    X_df = pd.DataFrame(data=X[0:, 0:], )

    # get accuracy with all features
    # individual = [1 for i in range(X.shape[1])]
    individual = [1 for i in range(len(X_df.columns))]

    # print("Train with all features: \t" +
    #       str(getFitness(individual, X_train_df, y_train)) + "\n")
    #
    # print("Test with all features: \t" +
    #       str(getFitness(individual, X_test_df, y_test)) + "\n")

    print("Accuracy with All features:")

    accuracy_train, accuracy_test = AccKFold(individual, X_df, y)

    logbook.record(Dataset="sat.all", avgTrain=accuracy_train[1], minFull=accuracy_test[0], avgFull=accuracy_test[1],
                   maxFull=accuracy_test[2])

    print("\nGA processing ...\n")
    # apply genetic algorithm
    hof = geneticAlgorithm(X_train_df, y_train, n_pop, n_gen)

    accuracy, individual, header = bestIndividual(hof, X, y)

    print("\nAccuracy with Subset features:")
    accuracy_train, accuracy_test = AccKFold(individual, X_df, y)

    logbook.record(min=accuracy_test[0], avg=accuracy_test[1], max=accuracy_test[2], AllFeature=len(individual),
                   SelectedFeature=individual.count(1), BestIndividual=individual)

    f.write(logbook.stream)

    print("\n")
    # print('Best Accuracy: \t' + str(accuracy))
    print('Number of Features in Subset: \t' + str(individual.count(1)) + "/" + str(len(individual)))
    print('Individual: \t\t' + str(individual))
    # print('Feature Subset\t: ' + str(header))
    #
    # print("Test with subset features: \t" +
    #       str(getFitness(individual, X_test_df, y_test)) + "\n")
