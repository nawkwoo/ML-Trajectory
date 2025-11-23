import numpy as np

X = np.load("preprocessed_data/X.npy")
y = np.load("preprocessed_data/y.npy")
q = np.load("preprocessed_data/quality.npy")

print(X.shape, y.shape, q.shape)