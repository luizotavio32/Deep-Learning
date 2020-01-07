import pandas as pd
import numpy as np
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

neural_net = Sequential()
neural_net.add(Dense(units = 8, activation = 'relu', 
                        kernel_initializer = 'normal', input_dim = 30))
neural_net.add(Dropout(0.2))
neural_net.add(Dense(units = 8, activation = 'relu', 
                       kernel_initializer = 'normal'))
neural_net.add(Dropout(0.2))
neural_net.add(Dense(units = 1, activation = 'sigmoid'))
neural_net.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

neural_net.fit(atributos, classes, batch_size = 10, epochs = 100)

novo = np.array([[15.80, 8, 118, 850, 0.10, 0.25, 0.08, 0.13, 0.178, 0.21, 0.05, 1099, 0.88, 4510, 145.2, 0.005, 0.04, 0.05, 0.015, 0.03, 0.007, 23.15, 16.78, 178.5, 2018, 0.14, 0.185, 0.84, 160, 0.363]])

pred = neural_net.predict(novo)
pred = (pred > 0.9)

#salva a rede
neural_net_json = neural_net.to_json()
with open('classificador_breast.json', 'w') as json_file:
    json_file.write(neural_net_json)
#salva os pesos
neural_net.save_weights('classificador_breast.h5')
