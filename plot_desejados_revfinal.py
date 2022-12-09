import matplotlib.pyplot as plt
import pandas as pd

#Formando dataframe
dados = pd.read_excel("PG43-TGA 2011temp.xlsx")
dados1 = pd.read_excel("PG43-TGB 2011temp.xlsx")
dados2 = pd.read_excel("PG43-TGC 2011temp.xlsx")
dados3 = pd.read_excel("PG43-TGD 2011temp.xlsx")

#Recebendo informações (dataXsensor)
data_inicial = '12/05/2011 12:21:00'
data_final = '13/05/2011 12:20:00'

#Transformando o eixo de datas em índices para facilitar o "procv"
dados.set_index('time', inplace=True)
dados1.set_index('time', inplace=True)
dados2.set_index('time', inplace=True)
dados3.set_index('time', inplace=True)

#print(f"\n Total de colunas {len(dados.columns)-1}.") 
indiceColunas = range(0,17) #eliminar coluna 0 pois contém as datas

for indiceColuna in indiceColunas: #eliminar primeira coluna de datas

    sensor = dados.columns[indiceColuna]
    sensor1 = dados1.columns[indiceColuna]
    sensor2 = dados2.columns[indiceColuna]
    sensor3 = dados3.columns[indiceColuna]
    #Subistituindo missing values pela mediana
    dados = pd.concat([dados.ffill(), dados.bfill()]).groupby(level=0).mean()
    dados1 = pd.concat([dados1.ffill(), dados1.bfill()]).groupby(level=0).mean()
    dados2 = pd.concat([dados2.ffill(), dados2.bfill()]).groupby(level=0).mean()
    dados3 = pd.concat([dados3.ffill(), dados3.bfill()]).groupby(level=0).mean()


    y = dados.loc[data_inicial:data_final]  #filtro de dados
    y1 = dados1.loc[data_inicial:data_final]  #filtro de dados
    y2 = dados2.loc[data_inicial:data_final]  #filtro de dados
    y3 = dados3.loc[data_inicial:data_final]  #filtro de dados
    #print(f"Data inicial {y.head(2)}, data final {y.head(-2)}.")
"""   
#Plotagem do gráfico de cada sensor
    plt.rcParams['figure.figsize'] = (10, 5)  
    plt.plot(y[f"{sensor}"])
    plt.title(f"Dados sensor {sensor} -TGC 2011") #adicionando o título
    plt.xlabel('02/09 a 03/09/2011', color="black") #definindo o nome do eixo X
    plt.ylabel("°C", color="black")  #definindo o nome do eixo Y
    plt.savefig(f'{sensor}_TGC_OIF_03_09_2011.png')
    plt.close("all")
    plt.show() #mostrando o gráfico
"""
   
#Plotagem todos os gráficos no mesmo plano
plt.rcParams['figure.figsize'] = (10, 5)   
plt.plot(y, )
plt.xlabel("Time (Mês/dia/hora)", color="red")
plt.ylabel("°C", color="red")
plt.legend(y.columns, bbox_to_anchor = (1, 1))
plt.suptitle("Sensores de Temperatura TGA OIF 13/05/2011")
plt.tight_layout()  
plt.savefig("Sensores de Temperatura TGA_OIF_13_05_2011.png")  
plt.show()  


#Plotagem todos os gráficos no mesmo plano
plt.rcParams['figure.figsize'] = (10, 5)   
plt.plot(y1, )
plt.xlabel("Time (Mês/dia/hora)", color="red")
plt.ylabel("°C", color="red")
plt.legend(y1.columns, bbox_to_anchor = (1, 1))
plt.suptitle("Sensores de Temperatura TGB OIF 13/05/2011")
plt.tight_layout()  
plt.savefig("Sensores de Temperatura TGB_OIF_13_05_2011.png")  
plt.show()  


#Plotagem todos os gráficos no mesmo plano
plt.rcParams['figure.figsize'] = (10, 5)   
plt.plot(y2, )
plt.xlabel("Time (Mês)/dia/hora)", color="red")
plt.ylabel("°C", color="red")
plt.legend(y2.columns, bbox_to_anchor = (1, 1))
plt.suptitle("Sensores de Temperatura TGC OIF 13/05/2011")
plt.tight_layout()  
plt.savefig("Sensores de Temperatura TGC_OIF_13_05_2011.png")  
plt.show()  

#Plotagem todos os gráficos no mesmo plano
plt.rcParams['figure.figsize'] = (10, 5)   
plt.plot(y3, )
plt.xlabel("Time (Mês/dia/hora)", color="red")
plt.ylabel("°C", color="red")
plt.legend(y3.columns, bbox_to_anchor = (1, 1))
plt.suptitle("Sensores de Temperatura TGD OIF 13/05/2011")
plt.tight_layout()  
plt.savefig("Sensores de Temperatura TGD_OIF_13_05_2011.png")  
plt.show()  

#Plotagem dos gráficos em subplot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 10), sharey=(True))
fig.suptitle("Sensores de Temperatura 13/05/2011",fontsize=20)
ax1.plot(y,)
ax1.set_title('TGA',fontsize=15)
ax2.plot(y1,)
ax2.set_title('TGB',fontsize=15)
ax3.plot(y2,)
ax3.set_title('TGC',fontsize=15)
ax4.plot(y3,)
ax4.set_title('TGD',fontsize=15)
#Coloca todas as figuras no mesmo formato
#for ax in fig.get_axes():
#    ax.label_outer()
fig.legend(y,bbox_to_anchor=(1.05,0.5), loc='center', borderaxespad=0.)  
fig.tight_layout()  
plt.savefig("Sensores de Temperatura_13_05_2011_subplot.png")  
plt.show() 

"""
#Plotagem dos gráficos em subplot
fig, axes = plt.subplots(6, 3, figsize=(20, 15))

for col, ax in zip(y.columns, axes.flatten()):
    ax.plot(y.index, y[col])
    ax.set_title(col)
    plt.subplots_adjust(wspace=.3, hspace=.5)
    
plt.xlabel("Time (Ano/dia/hora)", color="red")
plt.ylabel("°C", color="red")
fig.suptitle("Sensores de Temperatura")
fig.tight_layout()  
plt.savefig("Sensores de Temperatura TGD_OIF_03_09_2011_subplot.png")  
plt.show()    
"""