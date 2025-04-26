# data.py
import numpy as np
import pandas as pd

def generate_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'composition_A': np.random.uniform(0, 100, 100),
        'composition_B': np.random.uniform(0, 100, 100),
    })
    data['cost'] = 0.5 * data['composition_A'] + 0.8 * data['composition_B'] + np.random.normal(0, 5, 100)
    data['strength'] = 100 - 0.3 * data['composition_A'] + 0.4 * data['composition_B'] + np.random.normal(0, 5, 100)
    return data
