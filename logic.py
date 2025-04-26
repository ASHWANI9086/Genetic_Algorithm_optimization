# ga_nsga2.py
from deap import base, creator, tools, algorithms
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data import generate_data

# Load data
data = generate_data()
X = data[['composition_A', 'composition_B']].values
y = data[['cost', 'strength']].values
y[:, 1] = -y[:, 1]  # Maximize strength => minimize -strength

scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y)

# Define GA
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))  # Minimize both objectives
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 100)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(ind):
    cost = 0.5 * ind[0] + 0.8 * ind[1]
    strength = 100 - 0.3 * ind[0] + 0.4 * ind[1]
    return cost, -strength  # we want to maximize strength

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=0.6, low=0, up=100, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

def run_ga(pop_size=100, generations=50):
    pop = toolbox.population(n=pop_size)
    algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size, cxpb=0.7, mutpb=0.3, ngen=generations, verbose=False)
    return pop
