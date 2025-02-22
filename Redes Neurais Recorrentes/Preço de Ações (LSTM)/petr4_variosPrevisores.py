from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv('petr4-treinamento.csv')
base = base.dropna()
base_treinamento = base.iloc[:, 1:7].values

normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

normalizador_previsao = MinMaxScaler(feature_range=(0,1))
normalizador_previsao.fit_transform(base_treinamento[:,0:1])

previsores = []
preco_real = []
for i in range(90, 1242):
    previsores.append(base_treinamento_normalizada[i-90:i, 0:6])
    preco_real.append(base_treinamento_normalizada[i, 0])
previsores, preco_real = np.array(previsores), np.array(preco_real)

regressor = Sequential()
regressor.add(LSTM(units=100, return_sequences = True, input_shape = (previsores.shape[1], 6)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 1, activation= 'sigmoid'))

regressor.compile(optimizer = 'rmsprop', loss= 'mean_squared_error', metrics=['mean_absolute_error'])


es = EarlyStopping(monitor='loss', min_delta= 1e-10, patience=10, verbose=1)
lrr = ReduceLROnPlateau(monitor='loss', factor=0.2,patience=5, verbose=1)
mcp = ModelCheckpoint(filepath='pesos_variosPrev.h5', monitor='loss', save_best_only=True)

regressor.fit(previsores, preco_real, epochs=100, batch_size=32, callbacks=[es, lrr, mcp])

base_teste = pd.read_csv('petr4-teste.csv')
preco_real_teste = base_teste.iloc[:, 1:2].values

frames = [base, base_teste]
base_completa = pd.concat(frames)
base_completa = base_completa.drop('Date', axis = 1)


