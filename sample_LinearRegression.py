import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.arange(-5,5,0.1)
n = len(x)
newX = np.reshape(x,(n,1))

y = np.dot(newX, np.array([1]))
yNoise = 1 * np.random.normal(size=n)
y = y + yNoise

newY = np.reshape(y,(n,1))

linear_regression = LinearRegression()
 
reg = linear_regression.fit(newX, y)

# print(reg.score(newX, y))
# print(reg.coef_)
# print(reg.intercept_)

plt.scatter(x, y)
plt.plot(newX, reg.coef_*x, c="red")

plt.show()
