# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 12:42:18 2020

@author: Mansour Lo
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 14:28:35 2020

@author: chiss and Lo
"""

#modification du dossier par défaut
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

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

#constitution du nouvelle data frame
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
sel = SelectFwe(f_classif,alpha=0.0001)
sel.fit(newdata[newdata.columns[:-1]],newdata.V200)


#lister les variables sélectionnées
print(newdata.columns[:-1][sel.pvalues_ < 0.0001])

#construire la matrice X avec les var. sél. pour l'apprentissage
newX = sel.transform(x)
newX = pd.DataFrame(newX) 
newX['V200'] = newdata.V200
print(newX.shape) #(494021, 114)

#Nous separons les données de sorte a ce que toutes les modalités
#Soit plus ou moins presentes dans chaque base
data, DT = train_test_split(newX,train_size=290800,random_state=3)

#Nous recodons la variable cible pour la base apprentissage
datarecod  = pandas.get_dummies(data[['V200']])

#Nous rajoutons la variable cible pour la base apprentissage
data = pandas.concat([data[data.columns[:-1]],datarecod],axis=1)

#Nous recodons la variable cible pour la base test
datarecod  = pandas.get_dummies(DT[['V200']])

#Nous rajoutons la variable cible pour la base test
DT = pandas.concat([DT[DT.columns[:-1]],datarecod],axis=1)


#centrer et réduire les données
from sklearn.preprocessing import StandardScaler
std = StandardScaler()

#centrer et réduire
ZS = std.fit_transform(data)


# Nous testons ici la methode du elbow pour obtenir le K-optimal
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10), timings= True)
visualizer.fit(ZS)       
visualizer.show()

#Une fois obtenu nous appliquons la methode du k-means
# et nous entrainons un modele
kmeans = KMeans(n_clusters=3,n_init=10).fit(ZS)


#-----------------------------------------------------------------
m = DT.loc[DT.V200_m19==1]

test = kmeans.predict(m)
test = pd.Series(test)
print(test.value_counts())
#---------------------------------------------------------------

#Le code ci dessus permet de predire le ou les cluster(s) d'appartenance 
#de chaque modalité de Y grace a l'apprentissage sur la base apprentissage
#du k-means



#Permet la visualisation des clusters
from sklearn.decomposition import PCA

pca = PCA(2)
 
#Nous appliquons une PCA sur nos données
df = pca.fit_transform(ZS)
 

label = kmeans.fit_predict(df)

import matplotlib.pyplot as plt

#Nous cherchons les Centroids
centroids = kmeans.cluster_centers_
u_labels = np.unique(label)
 
#Nous plottons les resultats:
 
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] ,s = 7, label = i)
plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()