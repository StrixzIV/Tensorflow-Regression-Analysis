import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

#setup the base graph equations
n = 1000
x = np.random.rand(n, 1)
y = np.sin(2 * np.pi * x) + 0.4 * np.random.rand(n, 1)

#setup the 1 - 150 - 150 - 1 neural network model
inputs = keras.Input(shape = (1, ))
dense1 = layers.Dense(150, activation = 'relu')
dense2 = layers.Dense(150, activation = 'relu')
dense3 = layers.Dense(1)

output = dense3(dense2(dense1(inputs)))

#Train
model = keras.Model(inputs, output)
model.compile(optimizer = keras.optimizers.SGD(learning_rate = 0.1), loss = keras.losses.MeanSquaredError())

model.fit(x, y, epochs = 4000)

#plot
z = model.predict(x)

plt.plot(x, y, '.')
plt.plot(x, z, '.r')

plt.show()
