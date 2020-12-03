# Particle Swarm Optimization

import math
import operator
import random
from os.path import isfile

import numpy as np
import pandas as pd
from deap import base
from deap import creator
from deap import tools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
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

    if shuffle:
        ind = np.arange(X.shape[0])
        rng = check_random_state(random_state)
        rng.shuffle(ind)
        X = X[ind]
        y = y[ind]

    dataset = Bunch(data=X, target=y)

    return dataset


def generate(size, pmin, pmax, smin, smax):
    """
    khởi tạo một vị trí ngẫu nhiên và speed ngẫu nhiên cho một particle (hạt)
    :param size:
    :param pmin:
    :param pmax:
    :param smin:
    :param smax:
    :return:
    """
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part


def updateParticle(part, best, phi1, phi2):
    """
    đầu tiên sẽ tính toán speed, sau đó hạn chế các giá trị speed nằm giữa smin và smax,
    và cuối cùng là tính toán vị trí particle mới
    :param part:
    :param best:
    :param phi1:
    :param phi2:
    :return:
    """
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))


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

        clf = DecisionTreeClassifier()
        clf.fit(X_subset, y)

        # y_pred_ANN = clf.predict(X_test)
        # y_pred = clf.predict(X_subset)

        # return accuracy_score(y, y_pred_ANN)
        return (avg(cross_val_score(clf, X_subset, y, cv=5)),)
    else:
        return (0,)


def eaPSO(pop, toolbox, npop, ngen, stats=None,
          halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    if halloffame is not None:
        halloffame.update(pop)

    # record = stats.compile(pop)
    # logbook.record(gen=0, evals=len(pop), **record)
    # if verbose:
    #     print(logbook.stream)

    best = None

    # Begin the generational process
    for g in range(ngen):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:  # best fitness cho part
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:  # best fitness cho pop
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        halloffame.update(pop)

        # Gather all the fitnesses in one list and print the stats
        # Tổng hợp tất cả các fitness trong một list và show số liệu thống kê
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        # logbook.record(gen=g, evals=len(pop), **stats.compile(halloffame))
        if verbose:
            print(logbook.stream)

    return pop, logbook, best


def evolutionAlgorithm(X, y, n_population, n_generation):
    """
    Deap global variables
    Initialize variables to use eaSimple
    """
    # create individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # creator.create("Individual", list, fitness=creator.FitnessMax)
    creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
                   smin=None, smax=None, best=None)

    # create toolbox
    toolbox = base.Toolbox()
    # toolbox.register("attr_bool", random.randint, 0, 1)
    # toolbox.register("individual", tools.initRepeat,
    #                  creator.Individual, toolbox.attr_bool, len(X.columns))
    toolbox.register("particle", generate, size=2, pmin=-6, pmax=6, smin=-3, smax=3)
    # toolbox.register("population", tools.initRepeat, list,
    #                  toolbox.individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
    toolbox.register("evaluate", getFitness, X=X, y=y)
    # toolbox.register("evaluate", benchmarks.h1)
    # toolbox.register("mate", tools.cxOnePoint)
    # toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    # # toolbox.register("select", tools.selTournament, tournsize=3)
    # toolbox.register("select", tools.selNSGA2)

    # initialize parameters
    pop = toolbox.population(n=n_population)
    hof = tools.HallOfFame(n_population * n_generation)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # evolution algorithm
    pop, log, best = eaPSO(pop, toolbox,
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


def main():
    # GEN = 1000
    # best = None
    n_pop = 5
    n_gen = 5

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
    hof = evolutionAlgorithm(X_train_df, y_train, n_pop, n_gen)

    # select the best individual
    accuracy, individual, header = bestIndividual(hof, X, y)
    print('Best Accuracy: \t' + str(accuracy))
    print('Number of Features in Subset: \t' + str(individual.count(1)))
    print('Individual: \t\t' + str(individual))
    print('Feature Subset\t: ' + str(header))

    print("Test with subset features: \t" +
          str(getFitness(individual, X_test_df, y_test)) + "\n")


if __name__ == "__main__":
    main()
