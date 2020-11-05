from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
fromm sklearn.model_selection import train_test_split
from keras.optimizers import Adam

iris_data = load_iris()

features = iris_data.data
y = iris_data.target.reshape(-1, 1)

print(y)

#we have 3 classes so the labels will have 3 values
#but not 0,1 ,2 but now we use onehotencoder, hence first class: (1,0,0) second class: (0,1,0) third class:(0,0,1)

encoder=OneHotEncoder()
targets = encoder.fit_transform(labels).toarray()

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2)

model = Sequential()
model.add(Dense(10, input_dim=4, activation= 'sigmoid'))
model.add(Dense(3, activation='softmax'))
#input dim 4means here are 4 neurons
#we can define the loss fxn MSE or neg log likelihood
#optimizer will find the right adjustments for the weights: SGD, Adagrad, ADAM...

optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy',
              optimier=optimizer,
              metrics=['accuracy'])

model.fit(train_features, train_targets, epochs=10000, batch_size=20, vebose=2)
results = model.evelaute(test_features, tests_targets, use_multiprocessing= True)

print("training is finished... The loss and accuracy values are: ")
print(model.predict_proba(test_features))
print(results)


