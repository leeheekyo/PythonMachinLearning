import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5,5,0.1)
y = 1*(x) + 1

n = len(x)

yNoise = 1 * np.random.normal(size=n)
y = y + yNoise

x1 = x

newX1 = np.reshape(x1,(n,1))
baiasX = np.ones((n,1))
newX = np.append(baiasX,newX1,axis=1)

newXTranspose = np.transpose(newX)
newXTransposeDotX = newXTranspose.dot(newX)
newXTransposeDotXInv = np.linalg.inv(newXTransposeDotX)
newXTransposeDotY = newXTranspose.dot(y)
theta = newXTransposeDotXInv.dot(newXTransposeDotY)

print(theta)

plt.scatter(x, y)
plt.plot(x, theta[0] + theta[1]*x, c="red")

plt.show()
