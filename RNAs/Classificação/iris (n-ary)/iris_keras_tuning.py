import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)

def criar_rede(optimizer, kernel_initializer, activation, neurons, dropout):
    net = Sequential()
    net.add(Dense(units = neurons, activation = activation, 
                            kernel_initializer = kernel_initializer, input_dim = 4))
    net.add(Dropout(dropout))
    net.add(Dense(units = neurons, activation = activation, 
                            kernel_initializer = kernel_initializer))
    net.add(Dropout(dropout))
    net.add(Dense(units = 3, activation = 'softmax'))
    net.compile(optimizer = optimizer, 
                          loss = 'sparse_categorical_crossentropy', 
                          metrics = ['accuracy'])
    return net

net = KerasClassifier(build_fn = criar_rede)
parametros = {'batch_size': [10, 30],
              'epochs': [2000, 3000],
              'dropout': [0.2, 0.3],
              'optimizer': ['adam', 'sgd'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh', 'sigmoid'],
              'neurons': [4, 8, 16]}
grid_search = GridSearchCV(estimator = net,
                           param_grid = parametros,                           
                           cv = 10)
grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_