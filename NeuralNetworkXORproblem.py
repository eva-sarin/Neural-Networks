from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

print(x.shape)
print(y.shape)

model = Sequential()
model.add(Dense(4, input_dim=2, activation='sigmoid'))
model.add(Dense(4, input_dim=4, activation='sigmoid'))

print(model.weights)

model.compile(loss='mea_squared_error', optimizer='adam', metrics=['binary_accuracy'])
model.fit(x, y, epochs=100, verbose=2)

print("predictions after the training....")
print(model.predict(x))


