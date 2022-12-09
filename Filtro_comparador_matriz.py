# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 16:44:09 2021

@author: nayane
MATRIZ DE ACIONAMENTOS
"""

import pandas as pd

#Formando dataframe
dados = pd.read_excel("PG43-TGA 2011temp.xlsx")
dados1 = pd.read_excel("PG43-TGB 2011temp.xlsx")
dados2 = pd.read_excel("PG43-TGC 2011temp.xlsx")
dados3 = pd.read_excel("PG43-TGD 2011temp.xlsx")

dados.columns
dados1.columns
dados2.columns
dados3.columns

for idx, coluna in enumerate(dados.columns):
    if idx>0:
        filtro=dados[coluna]<400
        dados.loc[filtro,coluna]=0
        filtro=dados[coluna]>=400
        dados.loc[filtro,coluna]=1


for idx, coluna in enumerate(dados1.columns):
    if idx>0:
        filtro=dados1[coluna]<400
        dados1.loc[filtro,coluna]=0
        filtro=dados1[coluna]>=400
        dados1.loc[filtro,coluna]=1
        
for idx, coluna in enumerate(dados2.columns):
    if idx>0:
        filtro=dados2[coluna]<400
        dados2.loc[filtro,coluna]=0
        filtro=dados2[coluna]>=400
        dados2.loc[filtro,coluna]=1


for idx, coluna in enumerate(dados3.columns):
    if idx>0:
        filtro=dados3[coluna]<400
        dados3.loc[filtro,coluna]=0
        filtro=dados3[coluna]>=400
        dados3.loc[filtro,coluna]=1
       

dados.to_excel("PG43-TGA2011_matriz.xlsx", index= False)
dados1.to_excel("PG43-TGB2011_matriz.xlsx", index= False)
dados2.to_excel("PG43-TGC2011_matriz.xlsx",index= False)
dados3.to_excel("PG43-TGD2011_matriz.xlsx", index= False)

