import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense




##################### importando a base de dados / divisão da base entre treino e teste #####################

atributos = pd.read_csv('entradas-breast.csv')
classes = pd.read_csv('saidas-breast.csv')
atributos_treino, atributos_teste, classes_treino, classe_teste = train_test_split(atributos, classes, test_size = 0.25)




##################### criação da rede neural  #####################

neural_net = Sequential()

#1 camada oculta 
neural_net.add(Dense(units = 16, activation = 'relu', 
                    kernel_initializer = 'random_uniform', input_dim = 30))
# 2 camada oculta (se mostrou desnecessária nesse caso)
#neural_net.add(Dense(units = 16, activation = 'relu', 
#                   kernel_initializer = 'random_uniform'))
# camada de saída
neural_net.add(Dense(units = 1, activation = 'sigmoid'))




##################### treino da rede neural/ ajuste de parâmetros pra teste #####################

#otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
#neural_net.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

neural_net.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
neural_net.fit(atributos_treino, classes_treino, batch_size = 10, epochs = 100)



#obtendo pesos por neuronio
pesos0 = neural_net.layers[0].get_weights()
pesos1 = neural_net.layers[1].get_weights()
pesos2 = neural_net.layers[2].get_weights()

# testando a rede/ obtendo valroes de previsão
pred = neural_net.predict(atributos_teste)
pred = (pred > 0.5)



##################### avaliação da rede neural (acurácia, matriz de confusão, etc) #####################
precisao = accuracy_score(classe_teste, pred)
matriz = confusion_matrix(classe_teste, pred)
result = neural_net.evaluate(atributos_teste, classe_teste)




# dica: neuronios camada oculta = (atributos + neuronios de saida) / 2 -> neuronios camada oculta = (30 + 1)/2 = 15,5