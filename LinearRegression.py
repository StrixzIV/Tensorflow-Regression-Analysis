import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

n = 1000
x = np.random.rand(n)
y = 10 * x + 7 + 0.7 * np.random.rand(n)
plt.plot(x, y, '.')

w = np.random.rand()
b = np.random.rand()

w = tf.Variable(w)
b = tf.Variable(b)

lr = 0.1

for epoch in range(1000):
  with tf.GradientTape() as t:
    Y = w * x + b
    loss = tf.reduce_mean(pow(Y - y, 2))

  dw, db = t.gradient(loss, [w, b])
  w.assign_sub(lr * dw)
  b.assign_sub(lr * db)

  print(epoch, w.numpy(), b.numpy(), loss.numpy())

z = w * x + b

plt.plot(x, y, '.b')
plt.plot(x, z, '.r')
plt.show()