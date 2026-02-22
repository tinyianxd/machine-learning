import pandas as pd
import numpy as np
import random


dataset_path = 'data.csv'

# print(dataset_path)

dataset = pd.read_csv(dataset_path)
# print(dataset.describe())

# print(dataset.columns)


x_vars = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']


y_vars = 'diagnosis'

dataset_x = []
tmp = [[] for i in range(len(dataset[x_vars[0]]))]

for var in x_vars:
    for i, data in enumerate(dataset[var]):
        tmp[i].append(data)

for data in tmp:
    dataset_x.append(np.array(data))

#scale

for i, data in enumerate(dataset_x):
    _mean = np.mean(data)
    _std = np.std(data)
    dataset_x[i] -= _mean
    dataset_x[i] /= _std


dataset_y = dataset[y_vars]

def edit(var):
    if var == 'M':
        return 0
    return 1

dataset_y = list(map(edit, dataset_y))

def shuffle_data(data_x, data_y):
    tmp = list(zip(data_x, data_y))
    random.shuffle(tmp)
    return zip(*tmp)

dataset_x, dataset_y = shuffle_data(dataset_x, dataset_y)

# for i in range(5):
#     print(dataset_x[i], dataset_y[i])

test_size = 80

dataset_x_train = dataset_x[:-test_size]
dataset_x_test = dataset_x[-test_size:]

dataset_y_train = dataset_y[:-test_size]
dataset_y_test = dataset_y[-test_size:]

class Unit:
    def __init__(self, x_len):
        self.x_len = x_len
        self.weight = np.random.randn(x_len)
        self.bias = random.random()
        # self.relu_c = random.random()
        self.dw = np.zeros(self.weight.shape)
        self.db = 0
        # self.dc = 0
        self.gw = np.zeros(self.weight.shape)
        self.gb = 0
        # self.gc = 0
        self.mw = np.zeros(self.weight.shape)
        self.mb = 0
        # self.mc = 0
         
        return
    
    def update_theta(self, scale):
        self.weight -= self.mw * scale
        self.bias -= self.mb * scale
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def scale_gradient(self, mul):
        for i in range(len(self.gw)):
            self.gw[i] *= mul
        self.gb *= mul
        # self.gc *= mul
    
    def evaluate(self, X):
        return self.sigmoid(X.dot(self.weight) + self.bias)

    def compute(self, X):
        tmp = self.evaluate(X)
        dtmp = tmp * (1 - tmp)
        self.dw = X * dtmp
        self.db = dtmp
        return tmp, self.weight * dtmp
        '''
        if tmp > 0:
            self.dw = X * self.relu_c
            self.db = self.relu_c
            self.dc = tmp
            return tmp * self.relu_c, self.weight * self.relu_c # res, dw, db, dc, dx
        self.dw = np.zeros(self.weight.shape)
        self.db = 0
        self.dc = 0
        return 0, np.zeros(self.weight.shape)
        '''
    
    def update_gradient(self, mul):
        self.gw += self.dw * mul
        self.dw = 0
        self.gb += self.db * mul
        self.db = 0
        return
    
    def update_momentum(self, beta1):
        self.mw = self.mw * beta1 + (1 - beta1) * self.gw
        self.mb = self.mb * beta1 + (1 - beta1) * self.gb
        self.gw = np.zeros(self.weight.shape)
        self.gb = 0
        return

    def gradient_norm(self):
        res = self.gb**2
        for w in self.gw:
            res += w**2
        return res
    
class Layer:
    def __init__(self, X_size, unit_num):
        self.X_size = X_size
        self.unit_size = unit_num
        tmp = [Unit(X_size) for i in range(unit_num)]
        self.units = np.array(tmp)
        tmp = [[] for i in range(X_size)]
        self.X_dfdx = np.array(tmp)
        return
    
    def compute(self, X):
        y = []
        tmp = [[] for i in range(self.X_size)]
        for unit in self.units:
            res, dx = unit.compute(X)
            y.append(res)
            for i, _dx in enumerate(dx):
                tmp[i].append(_dx)
            # for _dx, _dfdx in zip(dx, tmp):
            #     _dfdx.append(_dx)
        
        # for i in tmp:
        #     i = np.array(i)
        self.X_dfdx = np.array(tmp)
                 
        return np.array(y)
    
    def evaluate(self, X):
        res = []
        for unit in self.units:
            res.append(unit.evaluate(X))
        return np.array(res)

class Model:
    def __init__(self):
        self.layers = []
        self.weight = np.array([])
        self.bias = random.random()
        self.learning_rate = 1e-3
        # self.threshold = 10
        self.momentum_weight = np.array([])
        self.momentum_bias = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.beta1t = self.beta1
        self.beta2t = self.beta2
        self.v = 0
        return
    
    def gradient_norm(self):
        norm = 0
        for layer in self.layers:
            for unit in layer.units:
                norm += unit.gradient_norm()
        return norm
        

    def evaluate(self, X):
        for layer in self.layers:
            X = layer.compute(X)
        return X.dot(self.weight) + self.bias
    
    def add(self, layer):
        self.layers.append(layer)
    
    def cost(self, x, y, type):
        if type == 'mse':
            return (x - y)**2
        if type == 'cross-entropy':
            return -(y * np.log(x) + (1 - y) * np.log(1 - x))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def update_gradient(self, mul):
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i].units)):
                self.layers[i].units[j].update_gradient(mul)
    
    def predict_derivative(self, data_X, data_y):
        dselfw = np.array([])
        dselfb = 0
        
        X = data_X
        for layer in self.layers:
            X = layer.compute(X)

        y_p = X.dot(self.weight) + self.bias
        y_p = self.sigmoid(y_p)
        dcz = (y_p - data_y)
        dselfw = dcz * X
        dselfb = dcz
        dx = dcz*self.weight

        dcost_df = [np.array([]) for i in range(len(self.layers))]
        dcost_df[-1] = dx

        for i in range(len(self.layers)):
            layer = -(i+1)
            if len(self.layers)+layer <= 0:
                break

            for fx in self.layers[layer].X_dfdx:
                dcost_df[layer-1] = np.append(dcost_df[layer-1], fx.dot(dcost_df[layer]))

        
        # for layer, dcdf_ in zip(self.layers, dcost_df):
        #     for unit, dcdf in zip(layer.units, dcdf_):
        #         unit.update_gradient(dcdf)
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i].units)):
                self.layers[i].units[j].update_gradient(dcost_df[i][j])

        return y_p, dselfw, dselfb
    
    def update_theta(self, scale):
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i].units)):
                self.layers[i].units[j].update_theta(scale)
                # unit.weight -= unit.mw * scale
                # unit.bias -= unit.mb * scale
    
    def update_momentum(self, gsw, gsb):
        self.momentum_weight = self.momentum_weight * self.beta1 + (1 - self.beta1) * gsw
        self.momentum_bias = self.momentum_bias * self.beta1 + (1 - self.beta1) * gsb
        # for layer in self.layers:
        #     for unit in layer.units:
        #         unit.update_momentum(self.beta1)
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i].units)):
                self.layers[i].units[j].update_momentum(self.beta1)

    def train_step(self, training_data, data_size):
        loss = 0
        accuracy = 0
        gsw = np.zeros(self.weight.shape)
        gsb = 0
        for X, y in training_data:
            prediction, dsw, dsb = self.predict_derivative(X, y)
            gsw += dsw
            gsb += dsb
            loss += self.cost(prediction, y, type='cross-entropy') / data_size
            if prediction > 0.5 and y == 1:
                accuracy += 1
            if prediction < 0.5 and y == 0:
                accuracy += 1
        accuracy /= data_size

        norm = gsb**2
        for w in gsw:
            norm += w**2
        norm += self.gradient_norm()

        self.v = self.v * self.beta2 + (1- self.beta2) * norm
        self.update_momentum(gsw, gsb)

        norm = np.sqrt(norm)
        scale = self.learning_rate / (np.sqrt(self.v / (1 - self.beta2t)) + 1e-8) / (1 - self.beta1t)
        self.beta1t *= self.beta1
        self.beta2t *= self.beta2

        self.weight -= self.momentum_weight * scale
        self.bias -= self.momentum_bias * scale
        self.update_theta(scale)
        return accuracy*100, loss

    
    def train(self, training_X, training_y, batch_size, epochs):
        self.weight = np.zeros(self.layers[-1].unit_size)
        self.momentum_weight = np.zeros(self.weight.shape)
        for e in range(epochs):
            average_accuracy = 0
            training_X, training_y = shuffle_data(training_X, training_y)
            batch_X = []
            batch_y = []
            cost = 0
            for X, y in zip(training_X, training_y):
                counter = 0
                if len(batch_X) < batch_size:
                    batch_X.append(X)
                    batch_y.append(y)
                else:
                    batch_X = np.array(batch_X)
                    batch_y = np.array(batch_y)
                    acc, loss = self.train_step(zip(batch_X, batch_y), batch_size)
                    average_accuracy += acc
                    cost += loss
                    counter += 1
                    batch_y = []

            print(f'step {e+1}, average accuracy/cost: {average_accuracy/counter}% / {cost/counter}')
        return
    
    def test(self, data_X, data_y):
        data_size = len(data_X)
        loss = 0
        accuracy = 0
        for X, y in zip(data_X, data_y):
            z = self.sigmoid(self.evaluate(X))
            if z > 0.5 and y == 1:
                accuracy += 1
            if z < 0.5 and y == 0:
                accuracy += 1
            loss += self.cost(self.sigmoid(self.evaluate(X)), y, type='cross-entropy') / data_size
        accuracy /= data_size
        return accuracy * 100, loss
    
    
model = Model()
model.add(Layer(len(x_vars), 10))
model.add(Layer(10, 10))
model.add(Layer(10, 10))
model.add(Layer(10, 10))
model.add(Layer(10, 10))



model.train(dataset_x_train, dataset_y_train, 40, 1000)

accuracy, cost = model.test(dataset_x_test, dataset_y_test)
print(f'final average accuracy/cost: {accuracy}% / {cost}')

'''
'''