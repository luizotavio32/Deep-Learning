import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

lbl_prev = LabelEncoder()
previsores[:, 0] = lbl_prev.fit_transform(previsores[:, 0])
previsores[:, 1] = lbl_prev.fit_transform(previsores[:, 1])
previsores[:, 3] = lbl_prev.fit_transform(previsores[:, 3])
previsores[:, 5] = lbl_prev.fit_transform(previsores[:, 5])
previsores[:, 8] = lbl_prev.fit_transform(previsores[:, 8])
previsores[:, 9] = lbl_prev.fit_transform(previsores[:, 9])
previsores[:, 10] = lbl_prev.fit_transform(previsores[:, 10])

