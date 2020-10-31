# Differential evolution

import random
import array
from os.path import isfile

import numpy as np
import pandas as pd

from deap import base, algorithms
from deap import benchmarks
from deap import creator
from deap import tools
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
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
    # # filename = "datasets\\Landsat7neighbour.txt"
    # # filename = "datasets\\landsatImg.txt"
    # filename = "datasets\\sat.all.txt"
    # # filename = "datasets\\ulc.txt"
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

    if shuffle:
        ind = np.arange(X.shape[0])
        rng = check_random_state(random_state)
        rng.shuffle(ind)
        X = X[ind]
        y = y[ind]

    dataset = Bunch(data=X, target=y)

    return dataset


def getFitness(individual2, X, y):
    """
    Feature subset fitness function
    """
    toolbox = base.Toolbox()
    individual = toolbox.clone(individual2)

    if individual.count(0) != len(individual):
        for i in range(len(individual)):
            if individual[i] >= 0.5:
                individual[i] = 1
            else:
                individual[i] = 0

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


def eaDE(pop, toolbox, CR, F, npop, ngen, ndim, stats=None,
         halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # Evaluate the individuals
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(pop)

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for g in range(1, ngen):
        for k, agent in enumerate(pop):
            a, b, c = toolbox.select(pop)
            y = toolbox.clone(agent)
            index = random.randrange(ndim)

            for i, value in enumerate(agent):
                if i == index or random.random() < CR:
                    y[i] = a[i] + F * (b[i] - c[i])
                    if y[i]>1:
                        y[i]=1
                    if y[i] <0:
                        y[i]=0

            y.fitness.values = toolbox.evaluate(y)

            if y.fitness.values[0] > agent.fitness.values[0]:
                pop[k] = y

        halloffame.update(pop)

        # Append the current generation statistics to the logbook
        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(pop), **record)
        if verbose:
            print(logbook.stream)

    # print("Best individual is ", halloffame[0], halloffame[0].fitness.values[0])

    return pop, logbook


def evolutionAlgorithm(X, y, n_population, n_generation, n_dimension):
    """
    Deap global variables
    Initialize variables to use eaDE
    """

    # create individual
    # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    # creator.create("Individual", list, fitness=creator.FitnessMax)
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

    # create toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.uniform, 0, 1)
    #toolbox.register("attr_float", random.uniform, -3, 3)
    # toolbox.register("individual", tools.initRepeat,
    #                  creator.Individual, toolbox.attr_bool, len(X.columns))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_dimension)
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)
    toolbox.register("select", tools.selRandom, k=3)
    toolbox.register("evaluate", getFitness, X=X, y=y)
    # toolbox.register("evaluate", benchmarks.sphere)
    # toolbox.register("mate", tools.cxOnePoint)
    # toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    # toolbox.register("select", tools.selTournament, tournsize=3)

    # Differential evolution parameters
    CR = 0.25
    F = 1
    n_pop = 300
    n_gen = 200

    # initialize parameters
    # pop = toolbox.population(n=MU);
    # hof = tools.HallOfFame(1)
    pop = toolbox.population(n=n_population)
    hof = tools.HallOfFame(n_population * n_generation)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # evolution algorithm
    pop, log = eaDE(pop, toolbox, CR=0.25, F=1, npop=n_population,
                    ngen=n_generation, ndim=n_dimension, stats=stats, halloffame=hof,
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


def main():

    # filename = "datasets\\Landsat7neighbour.txt"
    # filename = "datasets\\landsatImg.txt"
    filename = "datasets\\sat.all.txt"
    # filename = "datasets\\ulc.txt"

    available = isfile(filename)
    if available:
        f = open("DE-result.txt", "a")
    else:
        f = open("DE-result.txt", "w")
    f.write("\nDataset\tClassifier\tTrain\tTest\n")

    # Problem dimension
    # NDIM = 10

    # Differential evolution parameters
    CR = 0.25
    F = 1
    # MU = 300
    n_pop = 50
    n_gen = 50

    satimage = fetch_datasets(filename)
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

    # Problem dimension
    NDIM = len(X_df.columns)

    # apply genetic algorithm
    hof = evolutionAlgorithm(X_train_df, y_train, n_pop, n_gen, NDIM)

    # select the best individual
    accuracy, individual, header = bestIndividual(hof, X, y)

    if individual.count(0) != len(individual):
        # get index with value 0
        cols = [index for index in range(
            len(individual)) if individual[index] == 0]

        X_df = pd.DataFrame(data=X[0:, 0:], )  # values
        # get features subset
        X_parsed = X_df.drop(X_df.columns[cols], axis=1)
        X_subset = pd.get_dummies(X_parsed).to_numpy()

        # X_train, X_test, y_train, y_test = train_test_split(X_subset, y, stratify=y, random_state=0)

        # KFold Cross Validation approach
        kf = KFold(n_splits=10, shuffle=True)
        kf.split(X_subset)

        # Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to
        # this list
        accuracy_model_train = []
        accuracy_model_test = []

        # Iterate over each train-test split
        k = 1
        for train_index, test_index in kf.split(X):
            # Split train-test
            X_train, X_test = X_subset[train_index], X_subset[test_index]
            y_train, y_test = y[train_index], y[test_index]
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
        accuracy_train = avg(accuracy_model_train)
        accuracy_test = avg(accuracy_model_test)
        # print(accuracy_model)

        print("TRAIN: " + str(accuracy_train))
        print("TEST: " + str(accuracy_test) + "\n")

        f.write(filename + " DE" + "\t" + str(accuracy_train) + "\t" + str(accuracy_test) + "\n")

    print('Best Accuracy: \t' + str(accuracy))
    print('Number of Features in Subset: \t' + str(individual.count(1)) +"/"+ str(len(individual)))
    print('Individual: \t\t' + str(individual))
    # print('Feature Subset\t: ' + str(header))
    #
    # print("Test with subset features: \t" +
    #       str(getFitness(individual, X_test_df, y_test)) + "\n")


if __name__ == "__main__":
    main()
