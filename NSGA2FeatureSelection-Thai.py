import array
import random

import numpy as np
import pandas as pd
from deap import creator, base, tools, benchmarks
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from EvolutionaryAlgorithms.base_Thai import avg, getConfig, fetch_datasets, AccKFold


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

        # y_pred = clf.predict(X_subset)

        # return accuracy_score(y, y_pred_ANN)
        return (avg(cross_val_score(clf, X_subset, y, cv=5)),)
    else:
        return (0,)


def eaNSGA2(pop, toolbox, cxpb, mutpb, npop, ngen, stats=None,
            halloffame=None, verbose=__debug__):
    random.seed(None)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # Tạo quần thể có npop cá thể
    pop = toolbox.population(n=npop)

    # Evaluate the individuals with an invalid fitness
    # Định giá các cá thể chưa có fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(pop)

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    # Ở đây chỉ gán crowding distance cho các cá thể
    # mà không có lựa chọn thực sự nào được thực hiện
    pop, front0 = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= cxpb:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        # Đánh giá các cá thể có fitness không hợp lệ
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        # Chọn quần thể thế hệ tiếp theo
        pop, front = toolbox.select(pop + offspring, npop)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)
        print("Number of fronts: " + str(len(front)))

    # print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))

    return pop, front[0], logbook


def geneticAlgorithm(X, y, n_population, n_generation, n_dimension):
    """
    Deap global variables
    Initialize variables to use eaSimple
    """
    # create individual
    # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("FitnessMin", base.Fitness, weights=(1.0, 1.0))
    # creator.create("Individual", list, fitness=creator.FitnessMax)
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

    # create toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    # toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, len(X.columns))
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)
    # toolbox.register("evaluate", getFitness, X=X, y=y)
    toolbox.register("evaluate", benchmarks.rmsClass, X=X, y=y)
    toolbox.register("mate", tools.cxTwoPoint)
    # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=20.0)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    # toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=20.0, indpb=1.0 / n_dimension)
    # toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("select", tools.selNSGA2)

    # initialize parameters
    pop = toolbox.population(n=n_population)
    hof = tools.HallOfFame(n_population * n_generation)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # genetic algorithm
    pop, front0, log = eaNSGA2(pop, toolbox, cxpb=0.9, mutpb=0.05,
                               npop=n_population, ngen=n_generation, stats=stats, halloffame=hof,
                               verbose=True)

    # return hall of fame
    return hof, front0


def bestIndividual(hof, X, y):
    """
    Get the best individual
    """
    _individual = None
    minAccurcy = 2.0
    for individual in hof:
        # if (individual.fitness.values > maxAccurcy):
        if individual.fitness.values[0] < minAccurcy:
            minAccurcy = individual.fitness.values[0]
            _individual = individual

    _individualHeader = [list(X)[i] for i in range(
        len(_individual)) if _individual[i] == 1]
    return _individual.fitness.values, _individual, _individualHeader


if __name__ == '__main__':

    LIST_DATASET = getConfig("DATA-INPUT", "dataset")[:-1]
    # n_pop = getConfig("GA-CONFIG", "populationSIZE")
    # n_gen = getConfig("GA-CONFIG", "generationNUM")

    n_pop = 60  # :4
    n_gen = 50
    # n_pop = 4
    # n_gen = 5

    f = open("NSGA2-result.txt", "a")
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

        NDIM = len(X_df.columns)

        print("Accuracy with All features:")
        accuracy_train_full, accuracy_test_full = AccKFold(individual, X_df, y)

        print("\nNSGA2 processing ...\n")
        # apply genetic algorithm
        # hof, front0 = geneticAlgorithm(X_train_df, y_train, n_pop, n_gen)
        hof, front0 = geneticAlgorithm(X_df, y, n_pop, n_gen, NDIM)
        # select the best individual
        accuracy, individual, header = bestIndividual(front0, X, y)

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
