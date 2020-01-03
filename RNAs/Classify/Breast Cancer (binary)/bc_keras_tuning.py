import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV





##################### importando a base de dados / divisão da base entre treino e teste #####################

atributos = pd.read_csv('entradas-breast.csv')
classes = pd.read_csv('saidas-breast.csv')

def nova_rede(optimizer, loss, kernel_initializer, activation, neuronios):
    neural_net = Sequential()
    neural_net.add(Dense(units = neuronios, activation = activation, 
                        kernel_initializer = kernel_initializer, input_dim = 30))
    neural_net.add(Dropout(0.2))
    neural_net.add(Dense(units = neuronios, activation = activation, 
                       kernel_initializer = kernel_initializer))
    neural_net.add(Dropout(0.2))
    neural_net.add(Dense(units = 1, activation = 'sigmoid'))
    neural_net.compile(optimizer = optimizer, loss = loss, metrics = ['binary_accuracy'])
    return neural_net

net = KerasClassifier(build_fn = nova_rede)
params = {'batch_size': [10, 30],
          'epochs': [100, 150],
          'optimizer' : ['adam', 'sgd'],
          'loss' : ['binary_crossentropy', 'hinge'],
          'kernel_initializer' : ['random_uniform', 'normal'],
          'activation' : ['relu' , 'tanh'],
          'neuronios' : [16, 20]}

grid_search = GridSearchCV(estimator = net, param_grid = params, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(atributos, classes)
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_