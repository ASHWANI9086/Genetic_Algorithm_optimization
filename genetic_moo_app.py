# genetic_moo_app.py
# streamlit run genetic_moo_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import MinMaxScaler

# Step 1: Simulate Real-World-Like Industrial Data
np.random.seed(42)
data = pd.DataFrame({
    'composition_A': np.random.uniform(0, 100, 100),
    'composition_B': np.random.uniform(0, 100, 100),
})
data['cost'] = 0.5 * data['composition_A'] + 0.8 * data['composition_B'] + np.random.normal(0, 5, 100)
data['strength'] = 100 - 0.3 * data['composition_A'] + 0.4 * data['composition_B'] + np.random.normal(0, 5, 100)

# Step 2: Define Objective Function (Minimize Cost & Maximize Strength)
def evaluate(individual):
    A, B = individual
    cost = 0.5 * A + 0.8 * B
    strength = 100 - 0.3 * A + 0.4 * B
    return cost, -strength  # Minimizing both

# Step 3: NSGA-II Configuration
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))  # both to minimize
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 0, 100)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# Step 4: Run GA
def run_ga(pop_size=100, generations=50):
    pop = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()
    algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size, cxpb=0.6, mutpb=0.3,
                              ngen=generations, halloffame=hof, verbose=False)
    return hof

# Step 5: Streamlit App UI
st.title("🧬 Genetic Algorithm for Multi-Objective Optimization")
st.markdown("### Industrial Problem: Optimize Composition to Minimize Cost and Maximize Strength")

st.sidebar.header("GA Parameters")
pop_size = st.sidebar.slider("Population Size", 50, 300, 100, 10)
generations = st.sidebar.slider("Generations", 10, 200, 50, 10)

if st.button("Run Optimization"):
    hof = run_ga(pop_size, generations)

    result = pd.DataFrame([ind for ind in hof], columns=["composition_A", "composition_B"])
    result['cost'], strength = zip(*map(evaluate, hof))
    result['strength'] = [-s for s in strength]

    st.success("✅ Optimization Completed!")

    st.subheader("🔍 Pareto Front (Tradeoff Curve)")
    fig, ax = plt.subplots()
    ax.scatter(result["cost"], result["strength"], c="blue")
    ax.set_xlabel("Cost")
    ax.set_ylabel("Strength")
    ax.set_title("Pareto Front: Cost vs Strength")
    st.pyplot(fig)

    st.subheader("📋 Optimized Solutions")
    st.dataframe(result.round(2))
