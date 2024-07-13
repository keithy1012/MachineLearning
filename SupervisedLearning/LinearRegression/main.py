# Actual Implementation of Linear Regression
import pandas as pd
import matplotlib.pyplot as plt
import pandas as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data.csv")
print(df)

model = LinearRegression()
x = np.array(df.x.to_list()).reshape((-1, 1))
y = np.array(df.y)
model.fit(x, y)

b = model.intercept_
m = model.coef_
plt.scatter(x, y)
plt.plot(list(range(0, 60)), [m*x+b for x in range(0, 60)], color = "blue")
plt.show()

cost_function = RegressionCostFunctions()
actual = y
predicted = [m*x+b for x in range(0, 60)]
print(cost_function.MeanError(actual, predicted))