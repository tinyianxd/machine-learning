import os
import time
import pandas as pd
import numpy as np
import random


dataset_path = 'selfcode\\melb_data.csv'

dataset = pd.read_csv(dataset_path)

scale_vars = ['Rooms', 'Price', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Propertycount']
x_vars = ['Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Propertycount']
y_vars = 'Price'

for var in scale_vars:
    _mean = np.mean(dataset[var])
    _std = np.std(dataset[var])
    dataset[var] -= _mean
    dataset[var] /= _std

def shuffle_data(data_x, data_y):
    tmp = list(zip(data_x, data_y))
    random.shuffle(tmp)
    return zip(*tmp)


dataset_x = []
tmp = [[] for i in range(len(dataset[x_vars[0]]))]

for var in x_vars:
    for i, data in enumerate(dataset[var]):
        tmp[i].append(data)

for data in tmp:
    dataset_x.append(np.array(data))


dataset_y = dataset[y_vars]

dataset_x, dataset_y = shuffle_data(dataset_x, dataset_y)

dataset_x_train = dataset_x[:-3000]
dataset_x_test = dataset_x[-3000:]

dataset_y_train = dataset_y[:-3000]
dataset_y_test = dataset_y[-3000:]

class Unit:
    def __init__(self, x_len, relu_number):
        self.relu_number = relu_number
        self.learning_rate = 0.01
        self.x_len = x_len
        tmp = [np.random.randn(x_len) for i in range(relu_number)]
        self.weight = np.array(tmp)
        self.bias = np.zeros(relu_number)
        self.relu_c = np.random.randn(relu_number)
        return

    def test(self, test_x, test_y):
        return self.train_step(zip(test_x, test_y), len(test_x))
    
    def predict_derivative(self, X, y):
        res = 0
        drelu = [np.zeros(self.x_len) for i in range(self.relu_number)]
        dbias = [0]*self.relu_number
        dc = [0]*self.relu_number
        for i, (W, b, c) in enumerate(zip(self.weight, self.bias, self.relu_c)):
            tmp = W.dot(X) + b
            if tmp > 0:
                res += tmp * c
                drelu[i] = X * c
                dbias[i] = c
                dc[i] = tmp
        return res, np.array(drelu) * (res - y), np.array(dbias) * (res - y), np.array(dc) * (res - y)

    def cost(self, x, y):
        return (x-y)**2

    def train_step(self, training_data, data_size):
        loss = 0
        gweight = np.zeros(self.weight.shape)
        gbias = np.zeros(self.bias.shape)
        gc = np.zeros(self.relu_c.shape)
        for X, y in training_data:
            prediction, dweight, dbias, dc = self.predict_derivative(X, y)
            gweight += dweight / data_size
            gbias += dbias / data_size
            gc += dc / data_size
            loss += self.cost(prediction, y) / data_size
        self.weight -= self.learning_rate * gweight
        self.bias -= self.learning_rate * gbias
        self.relu_c -= self.learning_rate * gc
        return loss

    def train(self, training_X, training_y, batch_size, epochs):
        for e in range(epochs):
            total_loss = 0
            training_X, training_y = shuffle_data(training_X, training_y)
            batch_X = []
            batch_y = []
            for X, y in zip(training_X, training_y):
                if len(batch_X) < batch_size:
                    batch_X.append(X)
                    batch_y.append(y)
                else:
                    batch_X = np.array(batch_X)
                    batch_y = np.array(batch_y)
                    total_loss += self.train_step(zip(batch_X, batch_y), len(batch_X)) / (len(training_X)/len(batch_X))
                    batch_X = []
                    batch_y = []

            print(f'step {e+1}, average_loss: {total_loss}')


model = Unit(len(x_vars), 20)
model.train(dataset_x_train, dataset_y_train, 50, 50)

print(f'final average loss: {model.test(dataset_x_test, dataset_y_test)}')
