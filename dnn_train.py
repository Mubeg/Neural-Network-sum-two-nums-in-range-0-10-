import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
from tqdm import tqdm

LR = 1e-3

def initial_population():
    training_data = []
    for x in range(0, 10):
        for y in range(0, 10):
            ans = np.zeros(21, dtype = int)
            ans[x+y] = 1
            training_data.append([[x, y], ans])
    np.random.shuffle(training_data)
    return training_data

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

def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]
    if not model:
        model = neural_network_model(input_size = len(X[0]))
    
    model.fit({'input': X}, {'targets': y}, n_epoch=10000, snapshot_step=500, show_metric=True, run_id='math_learning')
    return model

training_data = initial_population()
model = train_model(training_data)

for _ in range(100):
    x = random.randint(0, 9)
    y = random.randint(0, 9)
    action = np.argmax(model.predict(np.array([x, y]).reshape(-1,len([x, y]),1))[0])
    print("X:", x, "Y:", y, "Ans:", action)
#model.save("'model_name'.model")
