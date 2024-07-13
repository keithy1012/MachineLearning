# Implementation of Linear Regression

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")
print(df)

# Calculates mean squared error
def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        actual_x = points.iloc[i].x
        actual_y = points.iloc[i].y
        predicted = m * actual_x + b
        total_error += (actual_y - predicted)**2
    total_error /= float(len(points))


def gradient_descent(m_curr, b_curr, points, l_rate):
    m_grad = 0
    b_grad = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].x
        y = points.iloc[i].y

        m_grad += -2/n * x * (y - (m_curr * x - b_curr))
        b_grad += -2/n * (y - (m_curr * x - b_curr))

    m = m_curr - m_grad * l_rate
    b = b_curr - b_grad * l_rate
    return m,b

m = 0
b = 0
L = 0.0001
generations = 1000

for generation in range(generations):
    m, b = gradient_descent(m, b, df, L)

print(m, b)
plt.scatter(df['x'], df['y'])
plt.plot(list(range(0, 60)), [m*x+b for x in range(0, 60)], color = "blue")
plt.show()