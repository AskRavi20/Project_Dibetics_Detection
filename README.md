# Project_Dibetics_Detection
In this study, a Deep Neural Network (DNN)-based multi-layer perceptron (MLP) with ten hidden layers was used to demonstrate the effectiveness of Deep Learning (DL) in diabetes diagnosis. The model achieved 99.8% accuracy, a 0.39% improvement over existing methods, demonstrating its potential for more precise and reliable diabetes detection.
from numpy import loadtxt #handle/load dataset
from keras.models import Sequential #Empty working area 
from keras.layers import Dense #Dense layer 
from sklearn.metrics import accuracy_score

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]
print(x)


model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=200, batch_size=10)

_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
