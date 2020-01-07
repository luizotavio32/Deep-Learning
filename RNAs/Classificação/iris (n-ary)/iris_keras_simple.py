import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

base = pd.read_csv('iris.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:, 4].values
lbl = LabelEncoder()
classe = lbl.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

x_treino, x_teste, y_treino, y_teste = train_test_split(previsores, classe_dummy, test_size=0.25)
#setosa 100
#virginica 010
#versicolor 001
net = Sequential()
net.add(Dense(units = 4, activation='relu', input_dim = 4))
net.add(Dense(units = 4, activation='relu'))
net.add(Dense(units = 3, activation='softmax'))
net.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['categorical_accuracy'])
net.fit(x_treino,y_treino, batch_size=10, epochs=1000)

resultado = net.evaluate(x_teste,y_teste)
prev = net.predict(x_teste)
y_teste2 = [np.argmax(x) for x in y_teste]
prev2 = [np.argmax(t) for t in y_teste]
matriz = confusion_matrix(prev2, y_teste2)