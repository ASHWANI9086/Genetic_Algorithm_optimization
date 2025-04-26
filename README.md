🎯 Genetic Algorithm for Multi-Objective Optimization (NSGA-II)
This project implements a Streamlit web app that applies the NSGA-II Genetic Algorithm to solve a multi-objective optimization problem in industrial material composition.

We optimize two conflicting objectives:

Minimize Cost 💵

Maximize Strength 💪

based on simulated industrial data.

🚀 Features
Streamlit Web Interface for easy interaction.

Simulated Industrial Data generation.

Pareto Front Visualization to show the trade-off between cost and strength.

Interactive Parameters: Choose population size and number of generations.

Top Optimal Combinations listing after optimization.

🛠️ Technologies Used
Python 3

Streamlit (for UI)

DEAP (for Genetic Algorithm)

Matplotlib (for plotting)

Scikit-learn (for scaling data)

📂 Project Structure
graphql
Copy
Edit
├── app.py                 # Streamlit main app
├── data.py                # Industrial data generation
├── logic.py (ga_nsga2.py)  # NSGA-II Genetic Algorithm logic
├── genetic_moo_app.py     # Another version of the Streamlit app
📈 How to Run
bash
Copy
Edit
# Install required packages
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
or
treamlit run genetic_moo_app.py
