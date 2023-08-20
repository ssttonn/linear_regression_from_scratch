import copy
import math

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("tvmarketing.csv")
print(dataset.describe(include="all"))
X_train = dataset.iloc[:, :-1].values
y_train = dataset.iloc[:, -1].values


def predict_row(X, w, b):
    return np.dot(X, w) + b


def compute_cost(X, y, w, b):
    total_cost = 0.
    m = X.shape[0]

    for i in range(m):
        error = predict_row(X[i], w, b) - y[i]
        total_cost += error ** 2
    total_cost /= 2 * m
    return total_cost


def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n, )
    dj_db = 0.
    for i in range(m):
        error = predict_row(X[i], w, b) - y[i]
        for j in range(n):
            dj_dw[j] += error * X[i, j]
        dj_db += error
    return dj_dw, dj_db


def gradient_descent(X, y, w, b, number_of_iterations, learning_rate):
    j_history = []
    w_temp = copy.deepcopy(w)
    b_temp = b
    for i in range(number_of_iterations):
        dj_dw, dj_db = compute_gradient(X, y, w_temp, b_temp)
        w_temp -= learning_rate * dj_dw
        b_temp -= learning_rate * dj_db
        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            j_history.append(compute_cost(X, y, w_temp, b_temp))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(number_of_iterations / 10) == 0:
            print(f"Iteration {i:4d}: Cost {j_history[-1]:8.20f}   ")
    return w_temp, b_temp, j_history


w_init = [0.1]
b_init = 1

w_result, b_result, J_history = gradient_descent(X_train, y_train, w_init, b_init, 150000, 34.51e-8)

y_hat = np.dot(X_train, w_result) + b_result
print(w_result)

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_history[:100])
ax2.plot(1000 + np.arange(len(J_history[1000:])), J_history[1000:])
ax1.set_title("Cost vs. iteration(start)")
ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')
ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')
ax2.set_xlabel('iteration step')
plt.show()

plt.scatter(X_train, y_train)
plt.plot(X_train, y_hat)
plt.show()
