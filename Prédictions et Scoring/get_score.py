# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 21:53:05 2020

@author: chiss and Lo
"""

import os
os.chdir("C:/Users/Mansour Lo/Desktop/Projet DataMining")

#lire les données d'apprentissage
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

#constitution du nouveau dataframe
newdata = pandas.concat([dataquali,testdata[colquanti]],axis=1)

#dimension
print(newdata.shape)

#liberer l'espace
del colquali, colquanti
del testdata, dataquali

#------------------------------------------------------------------------
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

#verifions la nouvelle dimmension
print(newX.shape) 
#---------------------------------------------------------------------

#centrer et réduire les données
from sklearn.preprocessing import StandardScaler
std = StandardScaler()

#centrer et réduire
ZS = std.fit_transform(newX)

#chargeons notre modele
#librairie chargement
import pickle

#ouverture en lecture binaire
f = open("mon_modele.sav","rb")

#et chargement
lr = pickle.load(f)

#fermeture du fichier
f.close()
#---------------------------------------------------------------------
#prediction proba
probas = lr.predict_proba(ZS)
print(probas)

#liste des classes
print(lr.classes_)

#la modalité positive est m16 qui est à la position 8

#score de 'positif' m16
score = probas[:,7]

#exportation du score
import numpy as np
np.savetxt("score.txt",score, fmt='%s',header="score")


#nombre de vrais positifs


#---------------------------------------------------------------------
# ytest=pandas.read_table(".txt",sep="\t", header=0)

# #transf. en 0/1 de Y_test
# pos = pandas.get_dummies(ytest).values
# #colonne de positif
# pos = pos[:,7]
# #nombre total de positif
# import numpy
# npos = numpy.sum(pos)
# print(npos) #3120
# #index pour tri selon le score croissant
# index = numpy.argsort(score)
# #inverser pour score décroissant
# index = index[::-1]
# #tri des individus (des valeurs 0/1)
# sort_pos = pos[index]
# #somme cumulée
# cpos = numpy.cumsum(sort_pos)
# #rappel
# rappel = cpos/npos
# #nb. obs ech.test
# n = ytest.shape[0] #1482063 observations
# #taille de cible
# taille = numpy.arange(start=1,stop=n+1,step=1)
# #passer en proportion
# taille = taille / n

# #graphique
# import matplotlib.pyplot as plt
# plt.title('Courbe de gain')
# plt.xlabel('Taille de cible')
# plt.ylabel('Rappel')
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.scatter(taille,taille,marker='.',color='blue')
# plt.scatter(taille,rappel,marker='.',color='red')
# plt.show()

# #vrai positif dans le 10 000 premiers

# val=int(10000*npos/n)

# prop_pos = rappel[val]
# print(prop_pos) #0.007051282051282051

# #on multiple par le nombre de positifs
# print(prop_pos * npos) #22

# # donc on peut tirer conclusion qu'il existe 22 positif parmis les 10 000 observations qui
# # présentent les scores les plus élevés dans une base de 1 482 063 obs.

#----------------------------------------------------------------------------------------------
