# Example of Gradient Descent for a MSE error function

import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
X = np.random.rand(20)
y = np.random.rand(20)
#print(X)
#print(y)

plt.scatter(X, y)
#plt.show()

# From sklearn (computational):
model = lm.LinearRegression()
model.fit(X.reshape(-1, 1), y)
m = model.coef_
b = model.intercept_

plt.scatter(X, y)
plt.scatter(X, m*X+b)
plt.show()

# Iteration Implementation:
m = 0
b = 0
lr = 0.001

for i in range(10000):
    y_predicted = m * X + b
    db = (1/20) * sum(y_predicted - y)
    dm = (1/20) * sum(y_predicted - y) * X
    m = m - lr * dm
    b = b - lr * db

plt.scatter(X, y)
plt.scatter(X, m*X+b)
plt.show()
