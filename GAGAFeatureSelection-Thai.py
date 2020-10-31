import random
import sys
from os.path import isfile

import numpy as np
import pandas as pd
from deap import creator, base, tools, algorithms, benchmarks
from deap.benchmarks.tools import hypervolume
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import Bunch, check_random_state

import coop_base

IND_SIZE = coop_base.IND_SIZE  # kích thước một cá thể trong loài
SPECIES_SIZE = coop_base.SPECIES_SIZE  # số cá thể trong loài
TARGET_SIZE = 30  # số cá thể trong target
NUM_SPECIES = 1  # số loài ban đầu


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
    # filename = "datasets\\Landsat7neighbour.txt"
    # filename = "datasets\\landsatImg.txt"
    filename = "datasets\\sat.all.txt"
    # filename = "datasets\\ulc.txt"

    available = isfile(filename)

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


def eaGAGA(toolbox, cxpb, mutpb, npop, ngen, stats=None,
           halloffame=None, verbose=__debug__):
    target_set = []

    logbook = tools.Logbook()
    logbook.header = "gen", "species", "evals", "std", "min", "avg", "max"

    # Tạo quần thể có MU cá thể
    # pop = toolbox.population(n=npop)
    target_set = toolbox.population(n=npop)

    # Tạo ngẫu nhiên target set (loài đích) có 30 cá thể
    # for i in range(len(schematas)):
    #     target_set.extend(toolbox.target_set(schematas[i], int(TARGET_SIZE / len(schematas))))

    # tạo NUM_SPECIES (loài), trong mỗi loài có IND_SIZE (cá thể)
    species = [toolbox.species() for _ in range(NUM_SPECIES)]

    # Init with random a representative for each species
    # Tạo ngẫu nhiên một cá thể ưu tú cho mỗi loài
    representatives = [random.choice(s) for s in species]

    g = 0
    # Begin the generational process
    while g < ngen:
        # Initialize a container for the next generation representatives
        # Tạo thùng chứa cá thể ưu tú của thế hệ tiếp theo
        next_repr = [None] * len(species)
        for i, s in enumerate(species):
            # Vary the species individuals
            # Biến đổi các cá thể trong loài
            s = algorithms.varAnd(s, toolbox, 0.6, 1.0)

            r = representatives[:i] + representatives[i + 1:]
            for ind in s:
                ind.fitness.values = toolbox.evaluate([ind] + r, target_set)

            record = stats.compile(s)
            logbook.record(gen=g, species=i, evals=len(s), **record)

            if verbose:
                print(logbook.stream)

            # Select the individuals
            # Lựa chọn các cá thể
            species[i] = toolbox.select(s, len(s))  # Tournament selection
            next_repr[i] = toolbox.get_best(s)[0]  # Best selection

            g += 1

        representatives = next_repr

    # return pop, logbook
    return representatives, logbook


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
    # toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("bit", random.randint, 0, 1)
    # toolbox.register("individual", tools.initRepeat,
    #                  creator.Individual, toolbox.attr_bool, len(X.columns))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.bit, IND_SIZE)
    toolbox.register("species", tools.initRepeat, list, toolbox.individual, SPECIES_SIZE)
    # toolbox.register("target_set", initTargetSet)

    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)
    # toolbox.register("evaluate", getFitness, X=X, y=y)
    # toolbox.register("evaluate", benchmarks.rmsClass, X=X, y=y)

    # toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mate", tools.cxTwoPoint)
    # toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1. / IND_SIZE)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("get_best", tools.selBest, k=1)
    toolbox.register("evaluate", coop_base.matchSetStrength)

    # initialize parameters
    # pop = toolbox.population(n=n_population)
    hof = tools.HallOfFame(n_population * n_generation)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)  # trung bình cộng (avg)
    stats.register("std", np.std)  # độ lệch chuẩn
    stats.register("min", np.min)
    stats.register("max", np.max)

    # genetic algorithm
    pop, log = eaGAGA(toolbox, cxpb=0.9, mutpb=0.2,
                      npop=n_population, ngen=n_generation, stats=stats, halloffame=hof,
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
    n_pop = 4
    n_gen = 50  # số thế hệ

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
    # print("Train with all features: \t" +
    #       str(getFitness(individual, X_train_df, y_train)) + "\n")
    #
    # print("Test with all features: \t" +
    #       str(getFitness(individual, X_test_df, y_test)) + "\n")

    # apply genetic algorithm
    hof = geneticAlgorithm(X_train_df, y_train, n_pop, n_gen)

    # select the best individual
    accuracy, individual, header = bestIndividual(hof, X, y)
    print('Best Accuracy: \t' + str(accuracy))
    print('Number of Features in Subset: \t' + str(individual.count(1)))
    print('Individual: \t\t' + str(individual))
    print('Feature Subset\t: ' + str(header))

    # print("Test with subset features: \t" +
    #       str(getFitness(individual, X_test_df, y_test)) + "\n")
