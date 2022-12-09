# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 19:10:43 2022

@author: nayane
"""

import pandas as pd
import scipy.stats as stats

dados = pd.read_excel(f'Dados EXP1_Rev1.xlsx', sheet_name=0) #coluna média
dados1 = pd.read_excel(f'Dados EXP1_Rev1.xlsx', sheet_name=1) #coluna máximo
dados2 = pd.read_excel(f'Dados EXP1_Rev1.xlsx', sheet_name=2) #coluna mínimo
dados3 = pd.read_excel(f'Dados EXP1_Rev1.xlsx', sheet_name=3) #coluna mediana
dados4 = pd.read_excel(f'Dados EXP1_Rev1.xlsx', sheet_name=4) #coluna desvio

shapiro_start,shapiro_p_valor = stats.shapiro(dados)
shapiro_start1,shapiro_p_valor1 = stats.shapiro(dados1)
shapiro_start2,shapiro_p_valor2 = stats.shapiro(dados2)
shapiro_start3,shapiro_p_valor3 = stats.shapiro(dados3)
shapiro_start4,shapiro_p_valor4 = stats.shapiro(dados4)

print("O valor da estatística de Shapiro-Wilk da média = "+ str(shapiro_start))
print("O valor da estatística de p de Shapiro-Wilk da média = "+ str(shapiro_p_valor))

print("O valor da estatística de Shapiro-Wilk do máximo = "+ str(shapiro_start1))
print("O valor da estatística de p de Shapiro-Wilk do máximo = "+ str(shapiro_p_valor1))

print("O valor da estatística de Shapiro-Wilk do mínimo = "+ str(shapiro_start2))
print("O valor da estatística de p de Shapiro-Wilk do mínimo = "+ str(shapiro_p_valor2))

print("O valor da estatística de Shapiro-Wilk da mediana = "+ str(shapiro_start3))
print("O valor da estatística de p de Shapiro-Wilk da mediana = "+ str(shapiro_p_valor3))

print("O valor da estatística de Shapiro-Wilk do desvio = "+ str(shapiro_start4))
print("O valor da estatística de p de Shapiro-Wilk do desvio = "+ str(shapiro_p_valor4))

if shapiro_p_valor > 0.05:
    print("Com 95% de confiança, os dados NÃO são similares a uma distribuição normal segundo o teste de Shapiro-Wilk")
else:
    print("Com 95% de confiança, os dados são similares a uma distribuição normal segundo o teste de Shapiro-Wilk")
    

if shapiro_p_valor1 > 0.05:
    print("Com 95% de confiança, os dados NÃO são similares a uma distribuição normal segundo o teste de Shapiro-Wilk")
else:
    print("Com 95% de confiança, os dados são similares a uma distribuição normal segundo o teste de Shapiro-Wilk")
    

if shapiro_p_valor2 > 0.05:
    print("Com 95% de confiança, os dados NÃO são similares a uma distribuição normal segundo o teste de Shapiro-Wilk")
else:
    print("Com 95% de confiança, os dados são similares a uma distribuição normal segundo o teste de Shapiro-Wilk")


if shapiro_p_valor3 > 0.05:
    print("Com 95% de confiança, os dados NÃO são similares a uma distribuição normal segundo o teste de Shapiro-Wilk")
else:
    print("Com 95% de confiança, os dados são similares a uma distribuição normal segundo o teste de Shapiro-Wilk")


if shapiro_p_valor4 > 0.05:
    print("Com 95% de confiança, os dados NÃO são similares a uma distribuição normal segundo o teste de Shapiro-Wilk")
else:
    print("Com 95% de confiança, os dados são similares a uma distribuição normal segundo o teste de Shapiro-Wilk")
