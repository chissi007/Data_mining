# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 15:08:33 2020

@author: chiss and Lo
"""

import os
os.chdir("C:/Users/Mansour Lo/Desktop/Projet DataMining")

#lire les données de deploiment
import pandas
testdata = pandas.read_table("DB.txt",sep="\t", header=0)

#convertissons des variables quali en quanti

#liste des variables quali
import numpy
colquali = [var for var in testdata.columns if testdata[var].dtype == numpy.object_]

#recodage en 0/1 ces variables 
dataquali = pandas.get_dummies(testdata[colquali])

#liste des variables quantitatives
colquanti = [var for var in testdata.columns if testdata[var].dtype != numpy.object_]

#constitution du nouvelle data frame
newdata = pandas.concat([dataquali,testdata[colquanti]],axis=1)
print(newdata.shape) #(1482063, 276)
#liberer l'espace
del colquali, colquanti
del testdata, dataquali
#--------------------------------------------------------------------
#selection des variables 

#chargeons le module contenat les infos necessaires

#librairie chargement
import pickle

#ouverture en lecture binaire
f = open("selection.sav","rb")

#et chargement
sel = pickle.load(f)

#fermeture du fichier
f.close()

#réduction de la base test aux var. sélectionnées
#en utilisant le filtre
newX=sel.transform(newdata)

#----------------------------------------------------------------------
#centrer et réduire les données
from sklearn.preprocessing import StandardScaler
std = StandardScaler()

#centrer et réduire
ZT = std.fit_transform(newX)

#chargeons notre modele de prediction
#librairie chargement du modèle
import pickle

#ouverture en lecture binaire de notre modele
f = open("mon_modele.sav","rb")

#et chargement
lr = pickle.load(f)

#fermeture du fichier
f.close()

#prediction grace au modele
pred = lr.predict(ZT)

#exportation de la prediction
import numpy as np

np.savetxt("prediction.txt",pred, fmt='%s',header="prediction")
