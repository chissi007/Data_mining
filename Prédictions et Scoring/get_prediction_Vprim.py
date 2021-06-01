# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 15:08:33 2020

@author: chiss and Lo
"""

import os
os.chdir("C:/Users/Mansour Lo/Desktop/Projet DataMining")

#lire les données de deploiement
import pandas 
testdata = pandas.read_table("DB.txt",sep="\t", header=0)
classe = pandas.read_table("classe.txt",sep="\t", header=0)

#converstissons des variables quali en quanti

#liste des variables quali
import numpy
colquali = [var for var in testdata.columns if testdata[var].dtype == numpy.object_]

#recodage en 0/1 ces variables 
dataquali = pandas.get_dummies(testdata[colquali])

#liste des variables quantitatives
colquanti = [var for var in testdata.columns if testdata[var].dtype != numpy.object_]

#constitution du nouvel data frame
newdata = pandas.concat([dataquali,testdata[colquanti]],axis=1)
print(newdata.shape) #(1482063, 276)
#liberer l'espace
del colquali, colquanti
del testdata, dataquali

#----------------------------------------------------------------------

#selection des variables 

#chargeons le module contenant les infos necessaires

#librairie chargement
import pickle

#ouverture en lecture binaire
f = open("selection_Vprim.sav","rb")

#et chargement
sel = pickle.load(f)

#fermeture du fichier
f.close()
#--------------------------------------------------------------------
#réduction de la base test aux var. sélectionnées
#en utilisant le filtre
newX=sel.transform(newdata)


#verifions la nouvelle dimmension
print(newX.shape)

#centrer et réduire les données
from sklearn.preprocessing import StandardScaler
std = StandardScaler()

#centrer et réduire
ZT = std.fit_transform(newX)

#chargeons notre modele de prediction
#librairie chargement du modèle
import pickle

#ouverture en lecture binaire
f = open("mon_modele_Vprim.sav","rb")

#et chargement
lr = pickle.load(f)

#fermeture du fichier
f.close()

#------------------------------------------------------------------------
#prediction
pred = lr.predict(ZT)

#exportation
import numpy as np

#La methode de regroupement est explicitée dans le rapport
Gr1 = set(['m4','m5','m6','m7','m10','m11','m12','m13','m16','m17','m18','m19','m20','m21','m22'])
Gr2 = set(['m1','m2','m9','m14','m23','m3','m8','m15'])

clsrgp1 = "Les classes regroupées sont : Grp1 : "
clsrgp2 = " et Grp2 : "

#Nousverifions que les variables contenus dans le fichier classe
#sont aussi contenu dans les groupe pré-construits
for i,row in classe.iterrows():
     if(row['classe'] in Gr1 ): 
        clsrgp1 += str(' '+row['classe'])

     if(row['classe'] in Gr2):
        clsrgp2 += str(' '+row['classe'])

clsrgp = clsrgp1+str('\n')+clsrgp2
clsrgp = clsrgp.split(',')


#Nous produisons deux fichiers le premier : La prediction
#Le second Les classes regroupées
np.savetxt("sorties1.txt",pred, fmt='%s',header="prediction")
np.savetxt("sorties2.txt",clsrgp,fmt='%s',header="Classe regroupes")


#----------------------------------------------------------------------
# #calcul le taux d'erreur 

# #import le Ytest

# #notre base test est la base initiale complete
# #donc on charge les Ycible reels

# #Nous comparons les Y  predits aux Y reels
# ytest= pandas.read_hdf('V200_Prim.h5',key='newX')

# print(numpy.mean(ytest != pred))

# #Taux d'erreur  =  0.00015586381955422948

#-----------------------------------------------------------------------

#Le code ci dessus permet le calcul du taux d'erreur