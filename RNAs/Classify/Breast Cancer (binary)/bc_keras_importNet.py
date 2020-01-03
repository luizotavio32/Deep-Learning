import numpy as np
from keras.models import model_from_json

arquivo = open('classificador_breast.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

net = model_from_json(estrutura_rede)
net.load_weights('classificador_breast.h5')

novo = np.array([[15.80, 8, 118, 850, 0.10, 0.25, 0.08, 0.13, 0.178, 0.21, 0.05, 1099, 0.88, 4510, 145.2, 0.005, 0.04, 0.05, 0.015, 0.03, 0.007, 23.15, 16.78, 178.5, 2018, 0.14, 0.185, 0.84, 160, 0.363]])

pred = net.predict(novo)
pred = (pred > 0.9)