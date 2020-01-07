import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json

# Carregamento e tratamento da base de dados
base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

# Criação da estrutura da rede neural e treinamento
net = Sequential()
net.add(Dense(units = 8, activation = 'relu', 
                        kernel_initializer = 'normal', input_dim = 4))
net.add(Dropout(0.1))
net.add(Dense(units = 8, activation = 'relu', 
                        kernel_initializer = 'normal'))
net.add(Dropout(0.1))
net.add(Dense(units = 3, activation = 'softmax'))
net.compile(optimizer = 'adam', 
                      loss = 'categorical_crossentropy', 
                      metrics = ['accuracy'])
net.fit(previsores, classe_dummy, 
                  batch_size = 10, epochs = 2000)

# Salvar o net
net_json = net.to_json()
with open("net_iris.json", "w") as json_file:
    json_file.write(net_json)
net.save_weights("net_iris.h5")

# Carregar o net
arquivo = open('net_iris.json', 'r')
estrutura_net = arquivo.read()
arquivo.close()
net_carregado = model_from_json(estrutura_net)
net_carregado.load_weights("net_iris.h5")

# Criar e classificar novo registro
novo = np.array([[3.2, 4.5, 0.9, 1.1]])
previsao = net.predict(novo)
previsao = (previsao > 0.5)
if previsao[0][0] == True and previsao[0][1] == False and previsao[0][2] == False:
    print('Iris setosa')
elif previsao[0][0] == False and previsao[0][1] == True and previsao[0][2] == False:
    print('Iris virginica')
elif previsao[0][0] == False and previsao[0][1] == False and previsao[0][2] == True:
    print('Iris versicolor')
    



