import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

#setup from the base graph equations
N = 100
X = np.random.rand(N)
Y = 5 * X + 10 + 0.4 * np.random.rand(N)
plt.plot(X, Y, '.b')

w = np.random.rand()
b = np.random.rand()

w = tf.Variable(w)
b = tf.Variable(b)

lr = 0.1

#Train the neural network
inputs = keras.Input(shape=(1,))
dense = layers.Dense(1)
outputs = dense(inputs)
model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1), loss=keras.losses.MeanSquaredError())

model.fit(X, Y, batch_size = N, epochs=1000)

#Plot
Z = model.predict(X)
plt.plot(X, Y, '.')
plt.plot(X, Z, '.r')
plt.show()
