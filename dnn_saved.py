import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

Const_Input_Size = 2
LR = 1e-3

def neural_network_model(input_size):

    network = input_data(shape=[input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = fully_connected(network, 256, activation='relu')
    network = fully_connected(network, 512, activation='relu')
    network = fully_connected(network, 256, activation='relu')
    network = fully_connected(network, 128, activation='relu')
    network = fully_connected(network, 128, activation='relu')
    network = fully_connected(network, 256, activation='relu')
    network = fully_connected(network, 512, activation='relu')
    network = fully_connected(network, 256, activation='relu')
    network = fully_connected(network, 128, activation='relu')

    network = fully_connected(network, 21, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

model = neural_network_model(input_size = Const_Input_Size)
model.load("best_sum_to_10.model")

hm_iterations = 10000
correct = 0
for _ in range(hm_iterations):
    x = random.randint(0, 9)
    y = random.randint(0, 9)
    prediction = np.argmax(model.predict(np.array([x, y]).reshape(-1,len([x, y]),1))[0])
    if prediction == x + y:
        correct += 1
    #print("X:", x, "Y:", y, "Ans:", prediction)
print(correct/hm_iterations)
