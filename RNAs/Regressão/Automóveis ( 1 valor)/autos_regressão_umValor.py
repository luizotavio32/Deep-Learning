import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score

base = pd.read_csv('autos.csv', encoding='ISO-8859-1')
base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)
base = base.drop('name', axis = 1)
base = base.drop('seller', axis = 1)
base = base.drop('offerType', axis = 1)
#base['name'].value_counts()

i1 = base.loc[base.price <= 10]

base = base.loc[base.price > 10]
base = base.loc[base.price < 350000]

base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts() # limousine
base['gearbox'].value_counts() # manual
base['model'].value_counts() # golf
base['fuelType'].value_counts() # gasolina = benzin
base['notRepairedDamage'].value_counts() # nao = nein

valores = {'vehicleType' : 'limousine',
            'gearbox': 'manuell',
            'model' : 'golf',
            'fuelType' : 'benzin', 
            'notRepairedDamage' : 'nein'}

base = base.fillna(value = valores)

previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values



# 0- 000
# 0- 010
# 3- 100

onehot = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(),[0,1,3,5,8,9,10])], remainder='passthrough')     
previsores = onehot.fit_transform(previsores).toarray()

def criar_rede():
    regressor = Sequential()
    regressor.add(Dense(units = 158, activation='relu', input_dim = 316))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 158, activation='relu'))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1, activation='linear'))
    regressor.compile(loss = 'mean_absolute_error', optimizer ='adam', metrics=['mean_absolute_error'])
    regressor.fit(previsores,preco_real, batch_size=300, epochs=100)
    return regressor

regressor = KerasRegressor(build_fn = criar_rede, epochs=100, batch_size = 300)

result = cross_val_score(estimator = regressor, X=previsores, y =preco_real, cv=10, scoring='neg_mean_absolute_error')

