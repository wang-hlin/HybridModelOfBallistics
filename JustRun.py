import operator
import random

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, gp, tools
from deapcalc import *
from prepdata import hybridmodeldataprep, hybridmodeltest

print(
    'Training data files should be in folder "data", files to fillin should be in the subfolder "fillin"'
)
file_name = str(
    input(
        "put file names for training here, if multiple files separate with comma(e.g. 6.5mm.csv, Remington.csv): "
    )
)
input_files = file_name.split(",")
input_files = [i.strip() for i in input_files]
raw_data = pd.concat(
    (pd.read_csv("./data/" + files) for files in input_files), ignore_index=True
)

if_fillin = str(input("Do you want to use this model to fill your template?(Y/N)"))
if if_fillin in ["y", "Y"]:
    fillin_name = str(
        input(
            "file names to fill in, if multiple files separate with comma(e.g. 6.5mm.csv, Remington.csv): "
        )
    )
    fillin_files = fillin_name.split(",")
    fillin_files = [i.strip() for i in fillin_files]

np_train, np_target, scaler = hybridmodeldataprep(raw_data)

pset = gp.PrimitiveSet("MAIN", 8)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)

pset.addPrimitive(add2, 2)
pset.addPrimitive(mul2, 2)
pset.addPrimitive(scala2, 1)
pset.addPrimitive(div2, 1)


pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))


pset.renameArguments(
    ARG0="D",
    ARG1="BC",
    ARG2="Weight",
    ARG3="boattail",
    ARG4="roundtip",
    ARG5="cannelure",
    ARG6="IV",
    ARG7="Veom",
)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evaluate(individual):
    func = gp.compile(individual, pset)
    sum_ = 0
    # Compute the output for each input then
    # sum to the rest the squared difference
    # between the target and the actual output.

    # a small constant number, to estimate the derivatives
    h = 1e-4

    for data, tar in zip(np_train, np_target):

        output = func(
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]
        )
        output_h1 = func(
            data[0] + h, data[1], data[2], data[3], data[4], data[5], data[6], data[7]
        )
        output_h0 = func(
            data[0] - h, data[1], data[2], data[3], data[4], data[5], data[6], data[7]
        )
        dv2 = (output_h1 + output_h0 - 2 * output) / (h ** 2)
        if data[0] == 0 and output == tar:
            sum_ += (tar - output) ** 2 / 1e8
        elif data[0] == 0 and output != tar:
            sum_ += 1e5
        else:
            sum_ += ((tar - output) ** 2) / 1e2
        if dv2 >= 0:
            sum_ += 1e3
        else:
            sum_ = sum_
    rmse = math.sqrt(sum_ / np_target.shape[0])

    # DEAP requires that evaluation function to always
    # return a tuple
    return (rmse,)


toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate(
    "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
)
toolbox.decorate(
    "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
)


def main():
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(2)
    pop, log = algorithms.eaSimple(
        pop, toolbox, 0.5, 0.6, 50, stats=mstats, halloffame=hof, verbose=True
    )
    return pop, log, hof


if __name__ == "__main__":
    pop, log, hof = main()

print("\nBest Symbolic Regression function:\n%s" % hof[0])
print("=================================================================")

if if_fillin in ["y", "Y"]:
    for files in fillin_files:
        fillin_data = pd.read_csv("./data/fillin/" + files)
        v_pred, v_real = hybridmodeltest(fillin_data, scaler, hof[0])
        df = pd.DataFrame(v_pred, columns=["V_predict"])
        fillin_data["V"] = df["V_predict"]
        fillin_data.to_csv("./data/fillin/" + files + "_predict.csv")
        print("file "+ files + "_predict.csv has been generated!")

print("=================================================================")
print("Script executed! You may save your function and close the script.")