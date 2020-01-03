import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score




##################### importando a base de dados / divisão da base entre treino e teste #####################

atributos = pd.read_csv('entradas-breast.csv')
classes = pd.read_csv('saidas-breast.csv')

def nova_rede():
    neural_net = Sequential()
    neural_net.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform', input_dim = 30))
    neural_net.add(Dense(units = 16, activation = 'relu', 
                       kernel_initializer = 'random_uniform'))
    neural_net.add(Dense(units = 1, activation = 'sigmoid'))
    otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
    neural_net.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    return neural_net

neural_net = KerasClassifier(build_fn = nova_rede, epochs = 100, batch_size = 10)

resultados = cross_val_score(estimator = neural_net, X = atributos, y = classes, cv = 10, scoring = 'accuracy')

media = resultados.mean()
desvio = resultados.std()

# dica: neuronios camada oculta = (atributos + neuronios de saida) / 2 -> neuronios camada oculta = (30 + 1)/2 = 15,5

# maior desvio padrão maior possibilade de overfitting