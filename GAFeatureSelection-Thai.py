import random

import numpy as np
import pandas as pd
from deap import creator, base, tools, algorithms
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from EvolutionaryAlgorithms.base_Thai import getConfig, avg, fetch_datasets, AccKFold


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
        # for col in cols:
        #     X_subset[col].values[:] = 0

        clf = DecisionTreeClassifier()
        clf.fit(X_subset, y)

        y_pred = clf.predict(X_subset)

        return accuracy_score(y, y_pred)
        # return (avg(cross_val_score(clf, X_subset, y, cv=5)),)
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

    LIST_DATASET = getConfig("DATA-INPUT", "dataset")[:-1]
    # n_pop = getConfig("GA-CONFIG", "populationSIZE")
    # n_gen = getConfig("GA-CONFIG", "generationNUM")

    n_pop = 5
    n_gen = 5

    f = open("GA-result.txt", "a")
    f.write("\n" + "-" * 100 + "\n")
    now = pd.datetime.datetime.now()
    f.write("Started: " + now.strftime('%d/%m/%Y %H:%M:%S\n\n'))
    # f.write("-" * 20 + "\n")

    logbook = tools.Logbook()
    logbook.header = ['Dataset', 'avgTrainFull', 'avgTrainSub', 'minTestFull', 'avgTestFull', 'maxTestFull',
                      'minTestSub', 'avgTestSub', 'maxTestSub', 'AllFeature',
                      'SelectedFeature', 'BestIndividual']

    i = 1
    for dataset in LIST_DATASET:
        print("\n")
        print("-" * 30 + str(i) + ".dataset:" + dataset + "-" * 30)

        satimage = fetch_datasets(dataset_name=dataset)
        X, y = satimage.data, satimage.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

        X_train_df = pd.DataFrame(data=X_train[0:, 0:], )  # values
        # index = X_train[1:, 0],  # 1st column as index
        # columns = X_train[0, 1:])  # 1st row as the column names

        X_test_df = pd.DataFrame(data=X_test[0:, 0:], )  # values

        X_df = pd.DataFrame(data=X[0:, 0:], )

        # get accuracy with all features
        individual = [1 for i in range(len(X_df.columns))]

        # print("Train with all features: \t" +
        #       str(getFitness(individual, X_train_df, y_train)) + "\n")
        #
        # print("Test with all features: \t" +
        #       str(getFitness(individual, X_test_df, y_test)) + "\n")

        print("Accuracy with All features:")
        accuracy_train_full, accuracy_test_full = AccKFold(individual, X_df, y)

        print("\nGA processing ...\n")
        # apply genetic algorithm
        # hof = geneticAlgorithm(X_train_df, y_train, n_pop, n_gen)
        hof = geneticAlgorithm(X_df, y, n_pop, n_gen)
        # select the best individual
        accuracy, individual, header = bestIndividual(hof, X, y)

        print("\nAccuracy with Subset features:")
        accuracy_train_sub, accuracy_test_sub = AccKFold(individual, X_df, y)

        logbook.record(Dataset=dataset, avgTrainFull=accuracy_train_full[1], avgTrainSub=accuracy_train_sub[1],
                       minTestFull=accuracy_test_full[0], avgTestFull=accuracy_test_full[1],
                       maxTestFull=accuracy_test_full[2], minTestSub=accuracy_test_sub[0],
                       avgTestSub=accuracy_test_sub[1], maxTestSub=accuracy_test_sub[2], AllFeature=len(individual),
                       SelectedFeature=individual.count(1), BestIndividual=individual)

        f.write(logbook.stream)
        f.write("\n")

        print("\n")
        # print('Best Accuracy: \t' + str(accuracy))
        print('Number of Features in Subset: \t' + str(individual.count(1)) + "/" + str(len(individual)))
        print('Individual: \t\t' + str(individual))
        # print('Feature Subset\t: ' + str(header))
        #
        # print("Test with subset features: \t" +
        #       str(getFitness(individual, X_test_df, y_test)) + "\n")
        i += 1

    f.close()
