import numpy as np
from scipy.optimize import minimize

import pandas as pd

df = pd.read_excel('data/optimization_scores_300.xlsx')

A = df['A'].tolist()
I = df['I'].tolist()
E = df['E'].tolist()
H = df['H'].tolist()

qualities = np.array([A,  # Quality 1
                      I,  # Quality 2
                      E])  # Quality 3
human_scores = np.array(H)


# Objective function to minimize (Mean Squared Error)
def objective(weights, qualities, human_scores):
    predicted_scores = np.dot(qualities.T, weights)
    return np.mean((predicted_scores - human_scores) ** 2)


initial_weights = np.ones(3) / 3

print('initial weights :',initial_weights)

# Constraints: The weights should sum up to 1
constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

# Optimization
result = minimize(objective, initial_weights, args=(qualities, human_scores), constraints=constraints)

# Extracting optimized weights
optimal_weights = result.x

print("Optimized weights:", optimal_weights)
