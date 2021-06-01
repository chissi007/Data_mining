# -*- coding: utf-8 -*-
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

#convertissons des variables quali en quanti

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
#-------------------------------------------------------------------
#separation des X et Y
x=newdata.iloc[:,:-1]

#sélection de variables - filtrage
from sklearn.feature_selection import SelectFwe,f_classif
sel = SelectFwe(f_classif,alpha=0.001)
sel.fit(newdata[newdata.columns[:-1]],newdata.V200)
#sel.fit(x,y)

#sauvegarde de la selection pour l'utiliser sur la base test

#librairie pour sauvegarde du modèle selection 
import pickle

#référence du fichier - ouverture en écriture binaire
f = open("selection_Vprim.sav","wb")

#sauvegarde dans le fichier référencé
pickle.dump(sel, f)

#fermeture du fuchier
f.close()

#-------------------------------------------------------------------

#construire la matrice X avec les var. sél. pour l'apprentissage
newX = sel.transform(x)
newX = pd.DataFrame(newX) 

#-----------------------------------------------------------------------------------------------------------

# # Gr1 = set(['m4','m5','m6','m7','m10','m11','m12','m13','m16','m17','m18','m19','m20','m21','m22'])

# # for i,row in newX.iterrows():
# #     print(i)
# #     if(row['V200'] in Gr1):
# #           newX.loc[i,'V200_Prim'] = 'Grp1'
# #     else:
# #           newX.loc[i,'V200_Prim'] = 'Grp2'

# # newX.to_hdf('dataVprim.h5', key='newX', mode='w')

#-------------------------------------------------------------------------------------------------------------

# #le code ci-dessus permet de creer la nouvelle colonne V200_Prim
# #Par soucis de temps nous le mettons en commentaire car prends 2H
# #Pour generer la nouvelle colonne

V200P =  pd.read_hdf('V200_Prim.h5',key='newX')

newX['V200_Prim'] = V200P

#une division 70/30 base apprentissage/base test         
data, DT = train_test_split(newX,train_size=345800,random_state=3)



#centrer et réduire les données pour la régression multinomiale
from sklearn.preprocessing import StandardScaler
std = StandardScaler()

#centrer et réduire
ZS = std.fit_transform(data[data.columns[:-1]])

#---------------------------------------------------------------------------------------------

#construction du modèle

from sklearn.linear_model import LogisticRegression

#régressions multinomiale avec lbfgs -- données centrées et réduite
lr = LogisticRegression(multi_class='multinomial',solver='lbfgs')
#apprentissage
lr.fit(ZS,data.V200_Prim)

#-------------------------------------------------------------------------
#librairie pour sauvegarde du modèle
import pickle

#référence du fichier - ouverture en écriture binaire
f = open("mon_modele_Vprim.sav","wb")

#sauvegarde dans le fichier référencé
pickle.dump(lr, f)

#fichier qu'il faut fermer
f.close()

#-----------------------------------------------------------------------
ytest = DT.V200_Prim

pred = lr.predict(std.fit_transform(DT.iloc[:,:-1]))

print(numpy.mean(ytest != pred))
#0.0003170940689915734