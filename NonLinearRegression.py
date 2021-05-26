import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

n = 100
x = np.random.rand(n, 1)
y = np.sin(2 * np.pi * x) + 0.4 * np.random.rand(n, 1)

plt.plot(x, y, '.')

w1 = tf.Variable(np.random.randn(1, 100))
w2 = tf.Variable(np.random.randn(100, 100))
w3 = tf.Variable(np.random.randn(100, 1))

b1 = tf.Variable(np.random.randn(100))
b2 = tf.Variable(np.random.randn(100))
b3 = tf.Variable(np.random.randn(1))

lr = 0.0001011

def relu(x):
  return tf.where(x>=0, x, 0)

setEpoch = 40000

for epoch in range(setEpoch):
  with tf.GradientTape() as t:
    Y = relu(x @ w1 + b1)
    Y = relu(Y @ w2 + b2)
    Y = Y @ w3 + b3

    loss = tf.reduce_mean(pow(Y - y, 2))

    dw1, db1, dw2, db2, dw3, db3 = t.gradient(loss, [w1, b1, w2, b2, w3, b3])
    
    w1.assign_sub(lr * dw1)
    w2.assign_sub(lr * dw2)
    w3.assign_sub(lr * dw3)

    b1.assign_sub(lr * db1)
    b2.assign_sub(lr * db2)
    b3.assign_sub(lr * db3)

    if epoch % 1000 == 0 or epoch == setEpoch:
      print(epoch, loss.numpy())
      
z = relu(x @ w1 + b1)
z = relu(z @ w2 + b2)
z = z @ w3 + b3

plt.plot(x, y, '.b')
plt.plot(x, z, '.r')
plt.show()