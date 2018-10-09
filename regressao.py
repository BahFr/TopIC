#1.Dados da base Diabetes
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
#itens da base
diabetes.keys()
print(diabetes.DESCR)
#pandas para manipular os dados em tabelas
import pandas
tabela = pandas.DataFrame(diabetes.data)
tabela.columns = diabetes.feature_names
print(tabela.head(10))
tabela['YDiabete'] = diabetes.target
tabela.head(10)

import matplotlib.pyplot as plt

plt.scatter(tabela.bmi, tabela.bp)
plt.xlabel('Índice de massa corporal')
plt.ylabel('Pressão arterial média')
plt.show()

#Métodos de correlação
print(tabela.corr())
plt.scatter(tabela.bmi, tabela.YDiabete)
plt.xlabel('Indice massa corporal')
plt.ylabel('YDiabete')
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 11:15:02 2018
@author: DELL
"""

plt.scatter(tabela.YDiabete, tabela.s5)
plt.xlabel('YDiabete')
plt.ylabel('S5')
plt.show()

#seleciona duas colunas
X = tabela[["bmi", "bp"]]
print(X)

#separa dados de treinamento do modelo linear e dados para validação do modelo
#inclui o modulo de regressão linear
from sklearn import linear_model

#separa em dois conjuntos, um para treinamento e outro para validação (20 últimos)
X_t = X[:-20]
X_v = X[-20:]
#print(X_t["RM"])
y_t = tabela["YDiabete"][:-20]
y_v = tabela["YDiabete"][-20:]
regr = linear_model.LinearRegression()
# treina o modelo
regr.fit(X_t, y_t)
# faz a predição
y_pred = regr.predict(X_v)
# coeficientes a
print('Coeficientes: \n', regr.coef_)
#intercepto b
print('Coeficientes: \n', regr.intercept_)
#y = 5.10*RM + -0.65*LSTAT + -1.24
#prediz manualmente os valores com base nos coeficientes encontrados na regressao
y_teste = 781.17*X_v["bmi"] - 410.14*X_v["bp"]- 152.86
#exibe o valor predito manualmente y_teste, que começa de 486
#exibe o valor real y_t
#exibe o valor predito pela regressão linear
#print(y_teste)
print(y_teste[422], y_t[0],y_pred[0])
#plota todos os valores de validação
plt.scatter(X_v["bmi"], y_v,  color='black')
plt.scatter(X_v["bmi"], y_pred, color='blue')
plt.legend(["Real", "Predito"])

