# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 14:28:35 2020

@author: chiss and Lo
"""

#modification du dossier par défaut
import os
import pandas as pd
from sklearn.model_selection import train_test_split

os.chdir("C:/Users/Mansour Lo/Desktop/Projet DataMining")

#lire les données d'apprentissage
import pandas
data = pandas.read_table("data_avec_etiquettes.txt",sep="\t", header=0)


#description 
data.info()

#converstissons des variables quali en quanti

#liste des variables quali
import numpy
colquali = [var for var in data.columns[:-1] if data[var].dtype == numpy.object_]
print(colquali)

#recodage en 0/1 ces variables 
dataquali = pandas.get_dummies(data[colquali])

#liste des variables quantitatives
colquanti = [var for var in data.columns[:-1] if data[var].dtype != numpy.object_]

#constitution du nouveau dataframe
newdata = pandas.concat([dataquali,data[colquanti]],axis=1)
print(newdata.info())
print(newdata.columns)

#rajoutons la variable cible
newdata['V200'] = data.V200
print(newdata.info())
print(newdata.shape) #(494021, 277)

#liberons un peu de l'espace
del data
del colquali
del colquanti
del dataquali

#separation des X et Y
x=newdata.iloc[:,:-1]

#sélection de variables - filtrage
from sklearn.feature_selection import SelectFwe,f_classif
sel = SelectFwe(f_classif,alpha=0.001)
sel.fit(newdata[newdata.columns[:-1]],newdata.V200)

#-------------------------------------------------------------------

#sauvegarde de la selection pour l'utiliser sur la base test

#librairie pour sauvegarde du modèle selection 
import pickle

#référence du fichier - ouverture en écriture binaire
f = open("selection.sav","wb")

#sauvegarde dans le fichier référencé
pickle.dump(sel, f)

#fermeture du fuchier
f.close()
#-------------------------------------------------------------------

#lister les variables sélectionnées
print(newdata.columns[:-1][sel.pvalues_ < 0.001])

#construire la matrice X avec les var. sél. pour l'apprentissage
newX = sel.transform(x)
newX = pd.DataFrame(newX) 
newX['V200'] = newdata.V200
print(newX.shape) #(494021, 113)

#on partage la base en deux : base apprentissage 70% et
#base test 30%
data, DT = train_test_split(newX,train_size=345800,random_state=3)


#centrer et réduire les données pour la régression multinomiale
from sklearn.preprocessing import StandardScaler
std = StandardScaler()

#centrer et réduire
ZS = std.fit_transform(data[data.columns[:-1]])

#--------------------------------------------------------------------

#construction du modèle

from sklearn.linear_model import LogisticRegression

#régressions multinomiale avec lbfgs -- données centrées et réduite
lr = LogisticRegression(multi_class='multinomial',solver='lbfgs')

#apprentissage
lr.fit(ZS,data.V200)

#librairie pour la sauvegarde du modèle
import pickle

#référence du fichier - ouverture en écriture binaire
f = open("mon_modele.sav","wb")

#sauvegarde dans le fichier référencé
pickle.dump(lr, f)

#fichier qu'il faut fermer
f.close()
#------------------------------------------------------------------

#Le calcul du taux d'erreur de notre modele sur la base test
ytest = DT.V200

pred = lr.predict(std.fit_transform(DT.iloc[:,:-1]))

print(numpy.mean(ytest != pred))
#0.000877068701466054