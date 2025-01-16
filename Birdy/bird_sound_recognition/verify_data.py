# verify_data.py
import numpy as np

# Load the features and labels
X = np.load('features.npy')
y = np.load('labels.npy')

# Print the shapes of the arrays
print(f'Features shape: {X.shape}')
print(f'Labels shape: {y.shape}')

# Print the first 5 entries as a sample
print(f'First 5 features: {X[:5]}')
print(f'First 5 labels: {y[:5]}')
