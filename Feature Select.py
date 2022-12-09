# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 05:18:16 2022

@author: nayane
"""

#Import de módulos
from sklearn import metrics
from statistics import mean
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
from pandas import read_excel
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Carregar o dataset EXP1
arquivo = 'Dados Normalizados EXP3_total.xlsx'
colunas = ['FALHA','MÉDIA','MÁXIMO','MÍNIMO','MEDIANA','DESVIO']

dados = read_excel(arquivo, names=colunas)

array = dados.values

X = array[:,1:6]
y = array[:,0]


r = []
"""
for e in range (1, 6):
    
    for i in range (0, 21):
        #Função para seleção de variáveis
        #For classification: chi2, f_classif, mutual_info_classif
        best_var = SelectKBest(score_func= chi2, k = e)
        #best_var = GenericUnivariateSelect(chi2, mode='percentile', param=4)

        #Executa a função de pontuação em (X,y) e obtém os recusos selecionados
        fit = best_var.fit(X,y)

        #Reduz X para os recursos selecionados
        features = fit.transform(X)
        r.append(features)


#Resultados
print('\nNúmero original de feature:', X.shape[1])
#print('\nNúmero reduzido de feature:', features.shape[1])
#print('\nFeatures selecionadas:\n\n', features)

"""

for e in range (0, 20):

    modelo = ExtraTreesClassifier()
    modelo.fit(X,y)
    r.append((modelo.feature_importances_)*100)
    

#Resultados
#print(r)
x = (dados.columns[1:6])
print(x,np.mean(r, axis=0)) 
r.append(np.mean(r, axis=0))
print(r)
"""
dataframe= pd.DataFrame(r)


with pd.ExcelWriter('teste_exp3_rev1.xlsx') as writer:
    dataframe.to_excel(writer)


labels = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12',
          'F13','F14','F15','F16','F17','F18','F19','F20']  

fig, ax = plt.subplots()
ax.set_xlabel("FOLDS", size=14)
ax.set_ylabel("PERCENTUAL DE IMPORTÂNCIA %", size=14)
ax.set_title("ORDEM DE IMPORTÂNCIA DAS CARACTERÍSTICAS", size=18)
ax.legend()
width = 0.35
ax.bar_label(labels - width/2, r0, width, padding=3)
ax.bar_label(labels + width/2, r1, width, padding=3)

fig.tight_layout()

plt.show()

plt.figure(figsize=(8, 6))
label = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
plt.bar(x, r[0], width=0.2, color=label)
plt.bar(x, r[1], width=0.2, color=label)
plt.xlabel("FOLDS", size=14) 
plt.ylabel("PERCENTUAL DE IMPORTÂNCIA %", size=14) 
plt.title("ORDEM DE IMPORTÂNCIA DAS CARACTERÍSTICAS", size=18)
plt.legend(x,bbox_to_anchor=(1.05,0.5), loc='center', borderaxespad=0.) 
plt.tight_layout()  
#plt.savefig("ORDEM DE IMPORTÂNCIA DAS CARACTERÍSTICAS_2METODO.png")  
plt.show()  


model = SelectFromModel(modelo).fit(X,y)
model.threshold_
model.get_support()
new = model.transform(X)
new.shape 
print(new)


#Plotagem dos gráficos em subplot
fig, ((ax1),(ax2)) = plt.subplots(2, 1, figsize=(18, 15))
fig.suptitle("ORDEM DE IMPORTÂNCIA DAS CARACTERÍSTICAS",fontsize=20)
ax1.bar(r,)
ax1.set_title('Importância em cada fold',fontsize=15)
ax2.bar(dados.columns[1:6],s,width=0.2, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'])
ax2.set_title('Média Geral',fontsize=15)
fig.legend(dados.columns[1:6],bbox_to_anchor=(1.05,0.5), loc='center', borderaxespad=0.)  
fig.tight_layout()  
plt.savefig("ORDEM DE IMPORTÂNCIA DAS CARACTERÍSTICAS_2METODOS_subplot.png")
plt.show() 

"""



