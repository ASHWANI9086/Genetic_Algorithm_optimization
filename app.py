# app.py
import streamlit as st
import matplotlib.pyplot as plt
from ga_nsga2 import run_ga
from data import generate_data

st.title("🎯 Genetic Algorithm for Multi-Objective Optimization")
st.markdown("Optimize **Cost** and **Strength** in industrial material composition using NSGA-II")

pop_size = st.slider("Population Size", 50, 200, 100)
generations = st.slider("Generations", 10, 100, 50)

if st.button("Run Genetic Algorithm"):
    with st.spinner("Optimizing..."):
        pop = run_ga(pop_size=pop_size, generations=generations)
        
        # Get results
        costs = [ind.fitness.values[0] for ind in pop]
        strengths = [-ind.fitness.values[1] for ind in pop]

        st.success("Optimization Done!")

        fig, ax = plt.subplots()
        ax.scatter(costs, strengths, c='blue', alpha=0.6)
        ax.set_xlabel("Cost")
        ax.set_ylabel("Strength")
        ax.set_title("Pareto Front (Cost vs Strength)")
        st.pyplot(fig)

        st.write("Top 5 Optimal Combinations:")
        for i, ind in enumerate(sorted(pop, key=lambda x: x.fitness.values[0])[:5]):
            st.write(f"#{i+1} → Composition A: {ind[0]:.2f}, B: {ind[1]:.2f} → Cost: {ind.fitness.values[0]:.2f}, Strength: {-ind.fitness.values[1]:.2f}")
