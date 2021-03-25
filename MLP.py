#References: Lab03-LinearRegression môn Cơ sở trí tuệ nhân tạo
#Link: https://colab.research.google.com/drive/1Ea2Q2dqLWIj-OApLSX1JOdoqpGv4HTIC?usp=sharing

import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
def gradient_descent(x, y, theta, alpha, m, iters):
    x_transpose = x.T
    for i in range(0, iters):
        h_theta = np.dot(x, theta)
        cost = np.sum((h_theta - y)**2/(2*m))
        dJ = np.dot(x_transpose, h_theta - y) / m
        theta = theta - alpha*dJ
    return theta

def generate_dataset(n, bias, variance):
    x = np.zeros(shape = (n,2))
    y = np.zeros(shape = n)
    for i in range(0, n):
        x[i][0] = 1
        x[i][1] = i
        y[i] = (i+bias) + random.uniform(0, 1) * variance
    return x, y

def plotting(x, y, w):
    plt.plot(x[: ,1], y, "x")
    plt.plot(x[: ,1], x * w, "r-")
    plt.show()

def visualize(model_function):
    x, y = generate_dataset(100, 5, 2)
    m, n = np.shape(x)
    iters = 5000
    alpha = 0.0005
    theta = np.ones(n)
    w = model_function(x, y, theta, alpha, m, iters)
    plotting(x, y, w)
    for i in range(0, iters):
        print('x = ', x)
        print('y = ', y)
        print('w = ', w)
visualize(gradient_descent)
