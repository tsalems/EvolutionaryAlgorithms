import random
import sys
from os.path import isfile

import numpy as np
import pandas as pd
from deap import creator, base, tools, algorithms, benchmarks
from deap.benchmarks.tools import hypervolume
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import Bunch, check_random_state


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


def eaNSGA2(pop, toolbox, cxpb, mutpb, npop, ngen, stats=None,
            halloffame=None, verbose=__debug__):
    random.seed(None)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # Tạo quần thể có MU cá thể
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
        print(len(front))

    # print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))

    return pop, front[0], logbook


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
    # toolbox.register("evaluate", getFitness, X=X, y=y)
    toolbox.register("evaluate", benchmarks.rmsClass, X=X, y=y)
    toolbox.register("mate", tools.cxTwoPoints)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
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
    pop, front0, log = eaNSGA2(pop, toolbox, cxpb=0.9, mutpb=0.2,
                               npop=n_population, ngen=n_generation, stats=stats, halloffame=hof,
                               verbose=True)

    # return hall of fame
    return hof, front0


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
        f = open("NSGA2-result.txt", "a")
    else:
        f = open("NSGA2-result.txt", "w")
    f.write("\nDataset\tClassifier\tTrain\tTest\n")

    n_pop = 60
    n_gen = 100

    satimage = fetch_datasets(filename=filename)
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

    # apply genetic algorithm
    hof, front0 = geneticAlgorithm(X_train_df, y_train, n_pop, n_gen)

    # select the best individual
    accuracy, individual, header = bestIndividual(front0, X, y)

    if individual.count(0) != len(individual):
        # get index with value 0
        cols = [index for index in range(
            len(individual)) if individual[index] == 0]

        # get features subset chay gi the cu
        X_parsed = X.drop(X.columns[cols], axis=1)
        X_subset = pd.get_dummies(X_parsed)

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
            X_train, X_test = X[train_index], X[test_index]
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

        f.write(filename + "NSGA2" + "\t" + str(accuracy_train) + "\t" + str(accuracy_test) + "\n")


    # print("Train with all features: \t" +
    #       str(getFitness(individual, X_train_df, y_train)) + "\n")
    #
    # print("Test with all features: \t" +
    #       str(getFitness(individual, X_test_df, y_test)) + "\n")



    # print('Best Accuracy: \t' + str(accuracy))
    # print('Number of Features in Subset: \t' + str(individual.count(1)))
    # print('Individual: \t\t' + str(individual))
    # print('Feature Subset\t: ' + str(header))

    # print(front0)

    # print("Test with subset features: \t" +
    #       str(getFitness(individual, X_test_df, y_test)) + "\n")
