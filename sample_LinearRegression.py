import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x1 = tf.Variable(tf.random.normal(shape=[1]))
x0 = tf.Variable(tf.random.normal(shape=[1]))

#Liear Regression
def Linear_Model(x):
    return x1*x +x0

#Loss function
def MES_LOSS(yPred, y):
    return tf.reduce_mean(tf.square(yPred-y))

#Optimizer
Optimizer = tf.optimizers.SGD(1e-2) 
def Training_Step(x,y):
    with tf.GradientTape() as tape:
        yPred = Linear_Model(x)
        loss = MES_LOSS(yPred, y)
    gradient = tape.gradient(loss, [x1,x0])
    Optimizer.apply_gradients(zip(gradient, [x1,x0]))


#Training Data
x = np.arange(-5,5,0.1)
y = 1*(x) + 1

n = len(x)

yNoise = 1 * np.random.normal(size=n)
y = y + yNoise

#Traning
for i in range(1000):
    Training_Step(x, y)

print(x1.numpy())
print(x0.numpy())

plt.scatter(x, y)
plt.plot(x, x0 + x1*x, c="red")

plt.show()
