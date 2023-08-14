import copy
import math

import numpy as np
import matplotlib.pyplot as plt

# Load our data set
x_train = np.array([1.0, 2.0])  # features
y_train = np.array([300.0, 500.0])  # target value


def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_square_cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        total_square_cost += (f_wb - y[i]) ** 2
    return total_square_cost / (2 * m)


def compute_gradients(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db


def gradient_descent(x, y, w, b, number_of_iterations, alpha, cost_function, gradient_function):
    w_final = copy.deepcopy(w)
    b_final = copy.deepcopy(b)
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []

    for i in range(number_of_iterations):
        dj_dw, dj_db = gradient_function(x, y, w_final, b_final)
        w_final -= alpha * dj_dw
        b_final -= alpha * dj_db

        if i < 100000:
            J_history.append(cost_function(x, y, w_final, b_final))
            p_history.append([w, b])
        if i % math.ceil(number_of_iterations / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w_final, b_final, J_history, p_history


w_init = 0
b_init = 0
tmp_alpha = 1.0e-2
iterations = 10000

w_result, b_result, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, iterations, tmp_alpha, compute_cost, compute_gradients)

print(f"(w,b) found by gradient descent: ({w_result:8.4f},{b_result:8.4f})")

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)")
ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')
ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')
ax2.set_xlabel('iteration step')
plt.show()
#
# y_hat = w_result * x_train + b_result
#
# plt.scatter(x_train, y_train, c="blue")
# plt.plot(x_train, y_hat)
# plt.show()
