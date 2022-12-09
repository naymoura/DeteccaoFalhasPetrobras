# -*- coding: utf-8 -*-
"""
Criado em: Thu Mar  4 21:00:01 2021

@author: Nayane Moura Neris

"""

import pandas as pd
import matplotlib.pyplot as plt


#arquivo = 'PG43-TGD 2012.csv'#,'PG43-TGB 2012.csv','PG43-TGC 2011.csv','PG43-TGA 2012.csv',
            #'PG43-TGC 2010.csv','PG43-TGB 2010.csv','PG43-TGD 2012.csv','PG43-TGA 2011.csv',
            #'PG43-TGB 2011.csv','PG43-TGD 2010.csv','PG43-TGD 2011.csv','PG43-TGA 2010.csv']

leitura = pd.read_csv(r"PG43-TGB.csv")

plt.rcParams["figure.figsize"] = (12.5, 5)


for sensor in leitura.columns[1:]:
    plt.plot(leitura["time"], leitura[f"{sensor}"])
    plt.title(f"Dados sensor {sensor} -TGB") #adicionando o título
    plt.xlabel("Datas (Jan./Dez.2010-2012)", color="black") #definindo o nome do eixo X
    plt.ylabel(f"{sensor}", color="black")  #definindo o nome do eixo Y
    plt.show() #mostrando o gráfico
    #plt.savefig(f'{sensor}.png') #salvando o gráfico

    
# Arquivos para armazenamento dos resultados
saida = pd.ExcelWriter(leitura[:-4]+'_calculos.xlsx')

# Incluir ou remover colunas conforme necessário
frameSaida = pd.DataFrame(columns=['SENSOR',
                                   'MEDIA',
                                   'MEDIANA',
                                   'MINIMO',
                                   'MAXIMO',
                                   'DESV.PADRAO',
                                   'DADOS AUSENTES'])

""" O loop abaixo percorre a lista de arquivos, calcula estatísticas em 
    cada coluna e salva em nome da planilha_calculos.xlsx.
"""

print(f"\n Processando arquivo {leitura}.")
            
#leitura = pd.read_csv('../'+dados,sep=';')

# Itera pelas colunas do arquivo atual e realiza os cálculos
for col in leitura.columns[1:]: 
    
    line = {}
    line['SENSOR'] = col
    line['MEDIA'] = leitura[col].mean()
    line['MEDIANA'] = leitura[col].median()
    line['MINIMO'] = leitura[col].min()
    line['MAXIMO'] = leitura[col].max()
    line['DESV.PADRAO'] = leitura[col].std()
    line['DADOS AUSENTES'] = leitura[col].isnull().sum()
    frameSaida = frameSaida.append(line,
                                   ignore_index=True)
    
    
    
# Salva o resultados do arquivo atual em uma planilha do excel.        
# A mesma planilha é utilizada para todos os arquivos, basta mudar
# a aba pelo parâmetro sheet_name
frameSaida.to_excel(excel_writer=saida,index=False)
# Escreve no disco
saida.save()