#import
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

#create dataset
n = 1000
x = np.random.rand(n)
y = 10 * x + 7 + 0.7 * np.random.rand(n)
plt.plot(x, y, '.')

#create random weight & bias value
w = np.random.rand()
b = np.random.rand()

#set weight & bias value for tensorflow
w = tf.Variable(w)
b = tf.Variable(b)

#set learning rate
lr = 0.1

#doing the regression
for epoch in range(1000):
  with tf.GradientTape() as t:
    Y = w * x + b
    loss = tf.reduce_mean(pow(Y - y, 2))

  dw, db = t.gradient(loss, [w, b])
  w.assign_sub(lr * dw)
  b.assign_sub(lr * db)

  print(epoch, w.numpy(), b.numpy(), loss.numpy())

#plot the result
z = w * x + b

plt.plot(x, y, '.b')
plt.plot(x, z, '.r')
plt.show()
