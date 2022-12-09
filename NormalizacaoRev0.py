# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:55:54 2021

@author: nayane


import pandas as pd

# Carregar o dataset
dados = pd.read_excel('Featrues1.xlsx')

#Listando índicies das colunas
colunas = list(dados.columns)

#while para percorrer cada coluna e executar o que se deseja
x = 3
while x <= len(colunas):
     dados_minmax_norm = (dados() - dados.min()) / ( dados.max() - dados.min())
     x += 1

print(dados_minmax_norm)
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
import statistics as sta

# Carregar o dataset
dados = pd.read_excel('EXP2mediana.xlsx')
#dados1 = pd.read_excel('Featrues4.xlsx')
dados.index = pd.to_datetime(dados.index, format='%Y/%m/%d %H:%M')
#dados1.index = pd.to_datetime(dados1.index, format='%Y/%m/%d %H:%M')


# sem parâmetros escala fica em [0,1] (default)
scaler = MinMaxScaler([0,1]) 

# ':,2:': as colunas 'Datas' e 'FALHA' não devem sofrer mudança de escala
# calculando máximo e mínimo para fazer a mudança
scaler.fit(dados.iloc[:,2:])

# obtendo dados modificados. Processando todo o DataFrame de uma única vez
scaled = scaler.transform(dados.iloc[:,2:])

# criando novo dataframe: usamos os parênteses antes de fazer a atribuição para
# conseguir 'quebrar' um comando em várias linhas
frameSaida = (
    pd.concat(
        [dados.loc[:,["Datas","FALHA"]], pd.DataFrame(data=scaled,columns=dados.columns[2:])],
        axis=1 # usamos este argumento para informar que iremos concatenar na horizontal
        )
    )

# para gravar no disco, podemos usar o to_excel diretamente com o nome do arquivo
frameSaida.to_excel('Dados Normalizados EXP2_MEDIANA.xlsx', index=False)

