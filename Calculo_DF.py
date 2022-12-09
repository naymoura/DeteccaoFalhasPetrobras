# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 11:01:43 2021

@author: nayane
"""

import statistics as sta
import pandas as pd

# Criando o DF dados e substituindo missing values pela média dos vizinhos próximos

dados = pd.read_csv('PG43-TGD.csv', sep=';',index_col='time')
dados.index = pd.to_datetime(dados.index, format='%Y/%m/%d %H:%M')
datas = pd.read_excel("DADOS EXP2.xlsx", sheet_name=3)

dados = pd.concat([dados.ffill(), dados.bfill()]).groupby(level=0).mean()

datas_iniciais = datas["Data/Hora Inicio Evento"]
datas_finais = datas["Data/Hora Fim Evento"]


#Frame_1440pontos = pd.DataFrame
frameSaida = pd.DataFrame
saida = pd.ExcelWriter('TGD2mediana.xlsx')
#Criando Frame inicial
for sensor in dados.columns[0:]:
    frameSaida = pd.DataFrame(columns=["Datas",
                                        sensor + " - min",
                                        sensor + " - max",
                                        sensor + " - mean",
                                        sensor + " - pstdev",
                                        sensor + " - median",
                                        "FALHA 1 N FALHA 0  "])
    frameSaida["FALHA 1 N FALHA 0 "] = datas["FALHA 1 N FALHA 0 "]
    frameSaida["Datas"] = datas_finais

for e in range(len(datas_iniciais)):
    
    
    data_inicial = datas_iniciais[e] 
    
    data_final = datas_finais[e] 
    
    print(f"Data inicial:{data_inicial}, data final:{data_final}")
    Frame_1440pontos = dados[data_inicial:data_final]

# Frame com dados pertencentes ao intervalo de data
    for sensor in dados.columns[0:]:

        #Cálculos
        minimo = min(Frame_1440pontos[sensor])
        maximo = max(Frame_1440pontos[sensor])
        media = sta.mean(Frame_1440pontos[sensor])
        desvio = sta.pstdev(Frame_1440pontos[sensor])
        mediana = sta.median(Frame_1440pontos[sensor])

        #Escrevendo xlsx
        line = {}
        line[sensor + " - min"] = minimo
        line[sensor + " - max"] = maximo
        line[sensor + " - mean"] = media
        line[sensor + " - pstdev"] = desvio
        line[sensor + " - median"] = mediana
        frameSaida = frameSaida.append(line, ignore_index=True)

        print(f"\nDados sobre o sensor: {sensor}\n\nMédia: {media} \nPonto máximo: {maximo}\nPonto mínimo: {minimo}\nDesvio Padrão: {desvio}\nMediana: {mediana}\nData: {data_final}")
frameSaida.to_excel(excel_writer=saida, index=False)
saida.save()

