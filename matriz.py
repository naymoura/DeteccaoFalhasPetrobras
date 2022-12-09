# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 01:05:59 2021

@author: nayan
"""

import pandas as pd


#TGA
#Lendo dataframe
dados1 = pd.read_excel(r"PG43-TGA2010_matriz.xlsx")
dados2 = pd.read_excel(r"PG43-TGA2011_matriz.xlsx")
dados3 = pd.read_excel(r"PG43-TGA2012_matriz.xlsx")
#TGB
dados4 = pd.read_excel(r"PG43-TGB2010_matriz.xlsx")
dados5 = pd.read_excel(r"PG43-TGB2011_matriz.xlsx")
dados6 = pd.read_excel(r"PG43-TGB2012_matriz.xlsx")
#TGC
dados7 = pd.read_excel(r"PG43-TGC2010_matriz.xlsx")
dados8 = pd.read_excel(r"PG43-TGC2011_matriz.xlsx")
dados9 = pd.read_excel(r"PG43-TGC2012_matriz.xlsx")
#TGD
dados10 = pd.read_excel(r"PG43-TGD2010_matriz.xlsx")
dados11 = pd.read_excel(r"PG43-TGD2011_matriz.xlsx")
dados12 = pd.read_excel(r"PG43-TGD2012_matriz.xlsx")


# Unindo as 3 planilhas
#2010
leitura1 = dados1.concat([dados4,dados7,dados10], ignore_index=True)
#2011
leitura2 = dados2.concat([dados5,dados8,dados11], ignore_index=True)
#2012
leitura3 = dados3.concat([dados6,dados9,dados12], ignore_index=True)


#salvando em CSV
#2010
leitura1.to_csv("MATRIZ 2010.csv", sep=";", index= False)
#2011
leitura2.to_csv("MATRIZ 2011.csv", sep=";", index= False)
#2012
leitura3.to_csv("MATRIZ 2012.csv", sep=";", index= False)


print(len(leitura1))
print(len(leitura2))
print(len(leitura3))
