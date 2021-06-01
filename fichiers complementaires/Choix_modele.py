# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 09:28:08 2020

@author: chiss
"""

#librairies et classes de calcul
import numpy
import os
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#changer de dossier
os.chdir("D:/dossier personelle/Lyon 2/M1/Data Mining/projet/logistic")

#charger les données
D = pandas.read_table("covertype.txt")
D.info()

#description
D.describe()

#distribution des classes
print(D.TYPE.value_counts())

#extraire un échantillon pour l'apprentissage, le reste pour le test
DS, DT = train_test_split(D,train_size=20000,random_state=0)
print(DS.shape)
print(DT.shape)

#régression multinomiale
lr = LogisticRegression(multi_class='multinomial',solver='newton-cg')

#apprentissage
%time lr.fit(DS.iloc[:,:-1],DS.TYPE)

#vérification
pred = lr.predict(DT.iloc[:,:-1])

#taux d'erreur
print(numpy.mean(DT.TYPE != pred))

#   warnings.warn("newton-cg failed to converge. Increase the "
# Wall time: 1min 54s
# 0.2829582967922255

#**************************************************************
#centrer et réduire les données pour la régression multinomiale
from sklearn.preprocessing import StandardScaler
std = StandardScaler()

#centrer et réduire
ZS = std.fit_transform(DS.iloc[:,:-1])

#régression multinomiale
lr = LogisticRegression(multi_class='multinomial',solver='newton-cg')

#apprentissage
%time lr.fit(ZS,DS.TYPE)

#vérification
pred = lr.predict(std.transform(DT.iloc[:,:-1]))

#taux d'erreur
print(numpy.mean(DT.TYPE != pred))

# Wall time: 31.6 s
# 0.27751991044754837

#*****************************************************************
#régressions multinomiale avec saga -- données centrées et réduite
lr = LogisticRegression(multi_class='multinomial',solver='saga')

#apprentissage
%time lr.fit(ZS,DS.TYPE)

#vérification
pred = lr.predict(std.transform(DT.iloc[:,:-1]))

#taux d'erreur
print(numpy.mean(DT.TYPE != pred))

#   warnings.warn("The max_iter was reached which means "
# Wall time: 18.7 s
# 0.2809012997939438

#*****************************************************************
#régressions multinomiale avec sag -- données centrées et réduite
lr = LogisticRegression(multi_class='multinomial',solver='sag')

#apprentissage
%time lr.fit(ZS,DS.TYPE)

#vérification
pred = lr.predict(std.transform(DT.iloc[:,:-1]))

#taux d'erreur
print(numpy.mean(DT.TYPE != pred))

#   warnings.warn("The max_iter was reached which means "
# Wall time: 7.9 s
# 0.2790706794150571

#régression multinomiale avec sag sans centrer reduit

lr = LogisticRegression(multi_class='multinomial',solver='sag')

#apprentissage
%time lr.fit(DS.iloc[:,:-1],DS.TYPE)

#vérification
pred = lr.predict(DT.iloc[:,:-1])

#taux d'erreur
print(numpy.mean(DT.TYPE != pred))

#  warnings.warn("The max_iter was reached which means "
# Wall time: 8.1 s
# 0.34223688619851267
#*****************************************************************
#régressions multinomiale avec lbfgs -- données centrées et réduite
lr = LogisticRegression(multi_class='multinomial',solver='lbfgs')

#apprentissage
%time lr.fit(ZS,DS.TYPE)

#vérification
pred = lr.predict(std.transform(DT.iloc[:,:-1]))

#taux d'erreur
print(numpy.mean(DT.TYPE != pred))

#   n_iter_i = _check_optimize_result(
# Wall time: 5.1 s
# 0.27779619687279417


#régression multinomiale avec lbfgs sans centrer reduit

lr = LogisticRegression(multi_class='multinomial',solver='lbfgs')

#apprentissage
%time lr.fit(DS.iloc[:,:-1],DS.TYPE)

#vérification
pred = lr.predict(DT.iloc[:,:-1])

#taux d'erreur
print(numpy.mean(DT.TYPE != pred))

#   n_iter_i = _check_optimize_result(
# Wall time: 4.9 s
# 0.38204708633683415
#*****************************************
#régressions binaires "ovr" avec liblinear
lr = LogisticRegression(multi_class='ovr',solver='liblinear')

#apprentissage
%time lr.fit(DS.iloc[:,:-1],DS.TYPE)

#vérification
pred = lr.predict(DT.iloc[:,:-1])

#taux d'erreur
print(numpy.mean(DT.TYPE != pred))
# Wall time: 6.12 s
# 0.28969255559595875

#régressions "ovr" avec liblinear donneés centrer reduit
lr = LogisticRegression(multi_class='ovr',solver='liblinear')

#apprentissage
%time lr.fit(ZS,DS.TYPE)

#vérification
pred = lr.predict(std.transform(DT.iloc[:,:-1]))

#taux d'erreur
print(numpy.mean(DT.TYPE != pred))
# Wall time: 9.17 s
# 0.28635038109701755

#régressions "ovr" avec lbfgs donneés centrer reduit
lr = LogisticRegression(multi_class='ovr',solver='lbfgs')

#apprentissage
%time lr.fit(ZS,DS.TYPE)

#vérification
pred = lr.predict(std.transform(DT.iloc[:,:-1]))

#taux d'erreur
print(numpy.mean(DT.TYPE != pred))

#   n_iter_i = _check_optimize_result(
# Wall time: 5.77 s
# 0.28625412647144804

#régressions "ovr" avec newton-cg donneés centrer reduit
lr = LogisticRegression(multi_class='ovr',solver='newton-cg')

#apprentissage
%time lr.fit(ZS,DS.TYPE)

#vérification
pred = lr.predict(std.transform(DT.iloc[:,:-1]))

#taux d'erreur
print(numpy.mean(DT.TYPE != pred))

# Wall time: 7.82 s
# 0.2862594739506463

#régressions binaires "ovr" avec newton-cg
lr = LogisticRegression(multi_class='ovr',solver='newton-cg')

#apprentissage
%time lr.fit(DS.iloc[:,:-1],DS.TYPE)

#vérification
pred = lr.predict(DT.iloc[:,:-1])

#taux d'erreur
print(numpy.mean(DT.TYPE != pred))

#   warnings.warn("newton-cg failed to converge. Increase the "
# Wall time: 53.8 s
# 0.2872968849151177