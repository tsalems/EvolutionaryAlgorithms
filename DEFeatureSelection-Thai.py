# Differential Evolution

import array
import random

import numpy as np
import pandas as pd
from deap import base
from deap import creator
from deap import tools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from EvolutionaryAlgorithms.base_Thai import avg, getConfig, fetch_datasets, AccKFold


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
        clf.fit(X_subset, y)

        # y_pred = clf.predict(X_subset)

        # return accuracy_score(y, y_pred_ANN)
        return (avg(cross_val_score(clf, X_subset, y, cv=5)),)
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
                    if y[i] > 1:
                        y[i] = 1
                    if y[i] < 0:
                        y[i] = 0

            y.fitness.values = toolbox.evaluate(y)

            if y.fitness.values[0] > agent.fitness.values[0]:
                pop[k] = y

        halloffame.update(pop)

        # Append the current generation statistics to the logbook
        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(pop), **record)
        if verbose:
            print(logbook.stream)

    print("Best individual is ", halloffame[0], halloffame[0].fitness.values[0])

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
    # toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("attr_float", random.uniform, 0, 1)
    # toolbox.register("individual", tools.initRepeat,
    #                  creator.Individual, toolbox.attr_bool, len(X.columns))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n_dimension)
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

        for i in range(len(individual)):
            if individual[i] >= 0.5:
                individual[i] = 1
            else:
                individual[i] = 0

        # if (individual.fitness.values > maxAccurcy):
        if individual.fitness.values[0] > maxAccurcy:
            maxAccurcy = individual.fitness.values[0]
            _individual = individual

    _individualHeader = [list(X)[i] for i in range(
        len(_individual)) if _individual[i] == 1]

    print('Individual: \t\t' + str(individual))

    return _individual.fitness.values, _individual, _individualHeader


def main():
    LIST_DATASET = getConfig("DATA-INPUT", "dataset")[:-1]
    # n_pop = getConfig("GA-CONFIG", "populationSIZE")
    # n_gen = getConfig("GA-CONFIG", "generationNUM")

    # Differential evolution parameters
    CR = 0.25
    F = 1
    # MU = 300
    n_pop = 50
    n_gen = 50

    f = open("DE-result.txt", "a")
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

        # Problem dimension
        # NDIM = 10
        NDIM = len(X_df.columns)

        print("Accuracy with All features:")
        accuracy_train_full, accuracy_test_full = AccKFold(individual, X_df, y)

        print("\nDE processing ...\n")
        # apply genetic algorithm
        hof = evolutionAlgorithm(X_train_df, y_train, n_pop, n_gen, NDIM)
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


if __name__ == "__main__":
    main()
