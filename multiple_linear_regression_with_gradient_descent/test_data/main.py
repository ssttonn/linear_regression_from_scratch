import copy
import math

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])


def row_predict(x, w, b):
    return np.dot(x, w) + b


def compute_cost(X, y, w, b):
    total_cost = 0
    m = X.shape[0]
    for i in range(m):
        f_wb = row_predict(X[i], w, b)
        total_cost = total_cost + (f_wb - y[i]) ** 2

    total_cost /= (2 * m)
    return total_cost


def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros([n, ])
    dj_db = 0.
    for i in range(m):
        error = row_predict(X[i], w, b) - y[i]
        for j in range(n):
            dj_dw[j] += error * X[i, j]
        dj_db = dj_db + error
    dj_dw /= m
    dj_db /= m
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

    return w_temp, b_temp, j_history  # return final w,b and J history for graphing


b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])

w_result, b_result, J_history = gradient_descent(X_train, y_train, np.zeros_like(w_init), b_init, 100000, 8e-7)
y_hat = np.dot(X_train, w_result) + b_result


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



