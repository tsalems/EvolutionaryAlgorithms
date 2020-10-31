#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

"""This is the base code for all four coevolution examples from *Potter, M.
and De Jong, K., 2001, Cooperative Coevolution: An Architecture for Evolving
Co-adapted Subcomponents.* section 4.2. It shows in a concrete manner how to
re-use initialization code in some other examples.
"""

# Base code của 4 ví dụ Đồng tiến hóa. Cụ thể để sử dụng lại code trong các ví dụ khác

import random

from deap import base
from deap import creator
from deap import tools

IND_SIZE = 64 # kích thước một cá thể trong loài
SPECIES_SIZE = 50 # số cá thể trong loài


def initTargetSet(schemata, size):
    """Initialize a target set with noisy string to match based on the
    schematas provided.
    Dựa trên schematas truyền vào, Tạo target set với nhiễu noisy string để match.
    """
    test_set = []
    for _ in range(size):
        test = list(random.randint(0, 1) for _ in range(len(schemata)))
        for i, x in enumerate(schemata):
            if x == "0":
                test[i] = 0
            elif x == "1":
                test[i] = 1
        test_set.append(test)
    return test_set


def matchStrength(x, y):
    """Compute the match strength for the individual *x* on the string *y*.
    Tính độ khớp cho từng cá thể *x* trên chuỗi *y*
    """
    return sum(xi == yi for xi, yi in zip(x, y))


def matchStrengthNoNoise(x, y, n):
    """Compute the match strength for the individual *x* on the string *y*
    excluding noise *n*.
    Tính độ khớp cho từng cá thể *x* trên chuỗi *y* không bao gồm nhiễu *n*
    """
    return sum(xi == yi for xi, yi, ni in zip(x, y, n) if ni != "#")


def matchSetStrength(match_set, target_set):
    """Compute the match strength of a set of strings on the target set of
    strings. The strength is the maximum of all match string on each target.
    Tính độ khớp của một tập các chuỗi trên target set.
    Strength là lớn nhất của tất cả các chuỗi khớp trên mỗi target.
    """
    sum = 0.0
    for t in target_set:
        sum += max(matchStrength(m, t) for m in match_set)
    return sum / len(target_set),


def matchSetStrengthNoNoise(match_set, target_set, noise):
    """Compute the match strength of a set of strings on the target set of
    strings. The strength is the maximum of all match string on each target
    excluding noise.
    Tính độ khớp của một tập các chuỗi trên target set.
    Độ khớp strength là lớn nhất của tất cả các chuỗi khớp trên mỗi target không bao gồm nhiễu.
    """
    sum = 0.0
    for t in target_set:
        sum += max(matchStrengthNoNoise(m, t, noise) for m in match_set)
    return sum / len(target_set),


def matchSetContribution(match_set, target_set, index):
    """Compute the contribution of the string at *index* in the match set.
    Tính sự đóng góp contribution của chuỗi tại *index* trong tập match set.
    """
    contribution = 0.0
    for t in target_set:
        match = -float("inf")
        id = -1
        for i, m in enumerate(match_set):
            v = matchStrength(m, t)
            if v > match:
                match = v
                id = i
        if id == index:
            contribution += match

    return contribution / len(target_set),


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("bit", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.bit, IND_SIZE)
toolbox.register("species", tools.initRepeat, list, toolbox.individual, SPECIES_SIZE)
toolbox.register("target_set", initTargetSet)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1. / IND_SIZE)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("get_best", tools.selBest, k=1)
toolbox.register("evaluate", matchSetStrength)
