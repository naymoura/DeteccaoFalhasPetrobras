# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 15:40:05 2022

@author: nayan
"""

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from statistics import mean
import pandas as pd
import numpy as np

# Carregar o dataset EXP1/EXP2
dados = pd.read_excel(f'Dados Normalizados EXP1_Rev1.xlsx', sheet_name=2)
#dados = pd.read_excel(f'Dados Normalizados EXP2_Rev1.xlsx', sheet_name=1)

dados.index

dados = dados.drop('Datas', axis=1)
#dados = dados.drop('Unnamed: 0', axis=1)

frameSaida = pd.DataFrame
#saida = pd.ExcelWriter('Exp1_resultados_Total_2folds.xlsx')
#saida = pd.ExcelWriter('Exp1_resultados_Desv_2folds.xlsx')
saida = pd.ExcelWriter('Exp1_resultados_DesvMed_2folds.xlsx')
#saida = pd.ExcelWriter('Exp1_resultados_Max_2folds.xlsx')
#saida = pd.ExcelWriter('Exp1_resultados_MaxMed_2folds.xlsx')
#saida = pd.ExcelWriter('Exp1_resultados_MaxMedDesv_2folds.xlsx')
#saida = pd.ExcelWriter('Exp1_resultados_MaxMedDesvMedian_2folds.xlsx')
#saida = pd.ExcelWriter('Exp1_resultados_MaxMedMin_2folds.xlsx')


#saida = pd.ExcelWriter('Exp2_resultados_Total_2folds.xlsx')
#saida = pd.ExcelWriter('Exp2_resultados_Max_2folds.xlsx')
#saida = pd.ExcelWriter('Exp2_resultados_MaxMed_2folds.xlsx')
#saida = pd.ExcelWriter('Exp2_resultados_MaxMedDesv_2folds.xlsx')
#saida = pd.ExcelWriter('Exp2_resultados_MaxMedDesvMedian_2folds.xlsx')
#saida = pd.ExcelWriter('Exp2_resultados_MaxMedMin_2folds.xlsx')

#saida = pd.ExcelWriter('Exp3_resultados_Total_2folds.xlsx')
#saida = pd.ExcelWriter('Exp3_resultados_Max_2folds.xlsx')
#saida = pd.ExcelWriter('Exp3_resultados_MaxMed_2folds.xlsx')
#saida = pd.ExcelWriter('Exp3_resultados_MaxMedDesv_2folds.xlsx')
#saida = pd.ExcelWriter('Exp3_resultados_MaxMedDesvMedian_2folds.xlsx')
#saida = pd.ExcelWriter('Exp3_resultados_MaxMedMin_2folds.xlsx')

Acuracia_xgb_mean = []
Acuracia_lgbm_mean = []
Acuracia_rfc_mean = []
Acuracia_etc_mean = []
Acuracia_rus_mean = []
Acuracia_mlp_mean = []

Acuracia_xgb1_mean = []
Acuracia_lgbm1_mean = []
Acuracia_rfc1_mean = []
Acuracia_etc1_mean = []
Acuracia_rus1_mean = []
Acuracia_mlp1_mean = []

for i in range (0,21):
    
    # Separa os dados do EXP1
    d0= dados.query('FALHA == 0').head(133)
    d1= dados.query('FALHA == 1').head(17)
    # Separa os dados do EXP2 
    #d0= dados.query('FALHA == 0').head(193)
    #d1= dados.query('FALHA == 1').head(32)
    # Separa os dados do EXP3
    #d0= dados.query('FALHA == 0').head(192)
    #d1= dados.query('FALHA == 1').head(17)
    #d2= dados.query('FALHA == 2').head(15)
    
    # Divide os dados de forma aleatória em dados menores formando 2 grupos
    result1 = np.array_split(d0, 2) # o split está retornando sempre os mesmos resultados
    result2 = np.array_split(d1, 2) 
    #result3 = np.array_split(d2, 2) 

    # Embaralhando conjuntos
    np.random.shuffle(result1)
    np.random.shuffle(result2)
    #np.random.shuffle(result3)

    C1 = []
    
    Acuracia_xgb = []
    Acuracia_lgbm = []
    Acuracia_rfc = []
    Acuracia_etc = []
    Acuracia_rus = []
    Acuracia_mlp = []
    
    Acuracia_xgb1 = []
    Acuracia_lgbm1 = []
    Acuracia_rfc1 = []
    Acuracia_etc1 = []
    Acuracia_rus1 = []
    Acuracia_mlp1 = []
    
    
    for e in range (0, 2):
        #print(result1[e],'\n')
        #print(result2[e],'\n')
        C = pd.concat([result1[e], result2[e]])
        #C = pd.concat([result1[e], result2[e], result3[e]])
        C1.append(C)
        #print(C,'\n')
    
    # print(C1[0],'\n')
    
    Train1 = pd.concat([C1[1]])
    #print(Train1,'\n')
    
    Train2 = pd.concat([C1[0]])
    #print(Train2,'\n')
    
    Train =  [Train1, Train2]
    
    print(i,f'Iteração:\n')
    
    for e in range(0,2):    
        # Dividindo a base em treino e validação
        X_train = Train[e].drop('FALHA', axis=1)
        y_train = Train[e]['FALHA']
        X_test = C1[e].drop('FALHA', axis=1)
        y_test = C1[e]['FALHA']
        

        """
          Alguns dos classificadores abaixo possuem fatores estocásticos,
          o que pode explicar a variação da média mesmo utilizando os mesmos conjuntos.
          Por exemplo o MLP tem este fator na inicialização dos pesos. O que significa que
          estamos variando o classificador ao invés do conjunto, no caso do MLP.

        """
        #xgb = XGBClassifier(learning_rate=0.1, n_estimators=1000, use_label_encoder=False, max_depth=8,
                            #objective="binary:logistic", subsample=0.75, colsample_bytree=0.85, seed=13, random_state=None)
        xgb = XGBClassifier(learning_rate=0.1, n_estimator=50, use_label_encoder=False)
        lgbm = LGBMClassifier(num_leaves=4, learning_rate=0.1, n_estimators=50, random_state=None)
        rfc = RandomForestClassifier(n_estimators=50, random_state=None)
        etc = ExtraTreesClassifier(n_estimators=50, random_state=None)
        rus = RUSBoostClassifier(n_estimators=50, algorithm='SAMME.R', random_state=None)
        mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=None)
        #mlp = MLPClassifier(learning_rate='adaptive', learning_rate_init= 0.001, solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=None, batch_size=20)
    
        #Treinando classificadores
        xgb.fit(X_train, y_train)
        lgbm.fit(X_train, y_train)
        rfc.fit(X_train, y_train)
        etc.fit(X_train, y_train)
        rus.fit(X_train, y_train)
        mlp.fit(X_train, y_train)
    
        #Predição de teste dos classificadores
        pred_xgb = xgb.predict(X_test)
        pred_lgbm = lgbm.predict(X_test)
        pred_rfc = rfc.predict(X_test)
        pred_etc = etc.predict(X_test)
        pred_rus = rus.predict(X_test)
        pred_mlp = mlp.predict(X_test)
        
        #Predição de treino dos classificadores
        pred_xgb1 = xgb.predict(X_train)
        pred_lgbm1 = lgbm.predict(X_train)
        pred_rfc1 = rfc.predict(X_train)
        pred_etc1 = etc.predict(X_train)
        pred_rus1 = rus.predict(X_train)
        pred_mlp1 = mlp.predict(X_train)
    
        #Printando resultados
        #print(f'\nResultados experimento {e}:\n')
        #print(f'Acúracia xgb: {accuracy_score(y_test, pred_xgb)}')
        #print(f'Acúracia lgbm: {accuracy_score(y_test, pred_lgbm)}')
        #print(f'Acúracia rfc: {accuracy_score(y_test, pred_rfc)}')
        #print(f'Acúracia etc: {accuracy_score(y_test, pred_etc)}')
        #print(f'Acúracia mlp: {accuracy_score(y_test, pred_mlp)}')
        #print(f'Acúracia rus: {accuracy_score(y_test, pred_rus)}\n')
           
        Acuracia_xgb.append(accuracy_score(y_test, pred_xgb))
        Acuracia_lgbm.append(accuracy_score(y_test, pred_lgbm))
        Acuracia_rfc.append(accuracy_score(y_test, pred_rfc))
        Acuracia_etc.append(accuracy_score(y_test, pred_etc))
        Acuracia_rus.append(accuracy_score(y_test, pred_rus))
        Acuracia_mlp.append(accuracy_score(y_test, pred_mlp))
        
        Acuracia_xgb1.append(accuracy_score(y_train, pred_xgb1))
        Acuracia_lgbm1.append(accuracy_score(y_train, pred_lgbm1))
        Acuracia_rfc1.append(accuracy_score(y_train, pred_rfc1))
        Acuracia_etc1.append(accuracy_score(y_train, pred_etc1))
        Acuracia_rus1.append(accuracy_score(y_train, pred_rus1))
        Acuracia_mlp1.append(accuracy_score(y_train, pred_mlp1))
    
    #Printando resultados das médias dos testes de 2 iterações
    print(f'A média dos testes dos 2 folds XGB:%.4f'%(mean(Acuracia_xgb)))
    print(f'A média dos testes dos 2 folds LGBM:%.4f'%(mean(Acuracia_lgbm)))
    print(f'A média dos testes dos 2 folds RFC:%.4f'%(mean(Acuracia_rfc)))
    print(f'A média dos testes dos 2 folds ETC:%.4f'%(mean(Acuracia_etc)))
    print(f'A média dos testes dos 2 folds RUS:%.4f'%(mean(Acuracia_rus)))
    print(f'A média dos testes dos 2 folds MLP:%.4f\n'%(mean(Acuracia_mlp)))
    #Printando resultados das médias sos treinos de 2 iterações
    print(f'A média dos treinos dos 2 folds XGB:%.4f'%(mean(Acuracia_xgb1)))
    print(f'A média dos treinos dos 2 folds LGBM:%.4f'%(mean(Acuracia_lgbm1)))
    print(f'A média dos treinos dos 2 folds RFC:%.4f'%(mean(Acuracia_rfc1)))
    print(f'A média dos treinos dos 2 folds ETC:%.4f'%(mean(Acuracia_etc1)))
    print(f'A média dos treinos dos 2 folds RUS:%.4f'%(mean(Acuracia_rus1)))
    print(f'A média dos treinos dos 2 folds MLP:%.4f\n'%(mean(Acuracia_mlp1)))
    
    #Calculando a médias dos testes   
    Acuracia_xgb_mean.append(mean(Acuracia_xgb))
    Acuracia_lgbm_mean.append(mean(Acuracia_lgbm))
    Acuracia_rfc_mean.append(mean(Acuracia_rfc))
    Acuracia_etc_mean.append(mean(Acuracia_etc))
    Acuracia_rus_mean.append(mean(Acuracia_rus))
    Acuracia_mlp_mean.append(mean(Acuracia_mlp))
    
    #Calculando a médias dos treinos
    Acuracia_xgb1_mean.append(mean(Acuracia_xgb1))
    Acuracia_lgbm1_mean.append(mean(Acuracia_lgbm1))
    Acuracia_rfc1_mean.append(mean(Acuracia_rfc1))
    Acuracia_etc1_mean.append(mean(Acuracia_etc1))
    Acuracia_rus1_mean.append(mean(Acuracia_rus1))
    Acuracia_mlp1_mean.append(mean(Acuracia_mlp1))
    
#Printando resultados das médias dos testes de todas as 20 iterações   
print(f'A média dos testes de todas as interações XGB:%.4f'%(mean(Acuracia_xgb_mean)))
print(f'A média dos testes de todas as interações LGBM:%.4f'%(mean(Acuracia_lgbm_mean)))
print(f'A média dos testes de todas as interações RFC:%.4f'%(mean(Acuracia_rfc_mean)))
print(f'A média dos testes de todas as interações ETC:%.4f'%(mean(Acuracia_etc_mean)))
print(f'A média dos testes de todas as interações RUS:%.4f'%(mean(Acuracia_rus_mean)))
print(f'A média dos testes de todas as interações MLP:%.4f\n'%(mean(Acuracia_mlp_mean)))
#Printando resultados das médias dos treinos de todas as 20 iterações
print(f'A média dos treinos de todas as interações XGB:%.4f'%(mean(Acuracia_xgb1_mean)))
print(f'A média dos treinos de todas as interações LGBM:%.4f'%(mean(Acuracia_lgbm1_mean)))
print(f'A média dos treinos de todas as interações RFC:%.4f'%(mean(Acuracia_rfc1_mean)))
print(f'A média dos treinos de todas as interações ETC:%.4f'%(mean(Acuracia_etc1_mean)))
print(f'A média dos treinos de todas as interações RUS:%.4f'%(mean(Acuracia_rus1_mean)))
print(f'A média dos treinos de todas as interações MLP:%.4f\n'%(mean(Acuracia_mlp1_mean)))




indiceColunas = range(0,6) #eliminar coluna 0 pois contém as datas
#Criando Frame inicial
for indiceColuna in indiceColunas: #eliminar primeira coluna de data
    frameSaida = pd.DataFrame(columns=["Classificador",
                                       "Média teste 2",
                                       "Média treino 2",
                                       "Média Geral teste 2",
                                       "Média Geral treino 2"])
#Escrevendo xlsx
    Classificador = ["XGB","LGBM","RFC","ETC","RUS","MLP"]
    Média_teste_2 = [mean(Acuracia_xgb), mean(Acuracia_lgbm),
                      mean(Acuracia_rfc), mean(Acuracia_etc),
                      mean(Acuracia_rus), mean(Acuracia_mlp)]
    
    Média_treino_2 = [mean(Acuracia_xgb1), mean(Acuracia_lgbm1),
                      mean(Acuracia_rfc1), mean(Acuracia_etc1),
                      mean(Acuracia_rus1), mean(Acuracia_mlp1)]
                          
    Média_geral_teste_2 = [mean(Acuracia_xgb_mean), mean(Acuracia_lgbm_mean),
                            mean(Acuracia_rfc_mean), mean(Acuracia_etc_mean),
                            mean(Acuracia_rus_mean), mean(Acuracia_mlp_mean)]
    
    Média_geral_treino_2 = [mean(Acuracia_xgb1_mean), mean(Acuracia_lgbm1_mean),
                            mean(Acuracia_rfc1_mean), mean(Acuracia_etc1_mean),
                            mean(Acuracia_rus1_mean), mean(Acuracia_mlp1_mean)]
    
 
    frameSaida = pd.DataFrame({'Classificador':Classificador,
                  'Média teste 2':Média_teste_2,
                  'Média treino 2':Média_treino_2,
                  'Média Geral teste 2':Média_geral_teste_2,
                  'Média Geral treino 2':Média_geral_treino_2,})
    
    frameSaida.to_excel(excel_writer=saida, index=False)
saida.save()


#Cria a matriz de confusão de teste
disp = metrics.plot_confusion_matrix(xgb, X_test, y_test)
disp.figure_.suptitle(f"Test Confusion Matrix XGBoost Featrues1 2 folds")
disp.figure_.savefig("Test Confusion Matrix XGB Featrues1_2 folds.png")
print(f"Test Confusion Matrix XGB:\n{disp.confusion_matrix}"'\n')
        
disp = metrics.plot_confusion_matrix(lgbm, X_test, y_test)
disp.figure_.suptitle(f"Test Confusion Matrix LGBM Featrues1 2 folds")
disp.figure_.savefig("Test Confusion Matrix LGBM Featrues1_2 folds.png")
print(f"Test Confusion Matrix LGBM:\n{disp.confusion_matrix}"'\n')
        
disp = metrics.plot_confusion_matrix(rfc, X_test, y_test)
disp.figure_.suptitle(f"Test Confusion Matrix RFC Featrues1 2 folds")
disp.figure_.savefig("Test Confusion Matrix RFC Featrues1_2 folds.png")
print(f"Test Confusion matrix RFC:\n{disp.confusion_matrix}"'\n')
        
disp = metrics.plot_confusion_matrix(etc, X_test, y_test)
disp.figure_.suptitle(f"Test Confusion Matrix ETC Featrues1 2 folds")
disp.figure_.savefig("Test Confusion Matrix ETC Featrues1_2 folds.png")
print(f"Test Confusion matrix ETC:\n{disp.confusion_matrix}"'\n')

disp = metrics.plot_confusion_matrix(rus, X_test, y_test)
disp.figure_.suptitle(f"Test Confusion Matrix RUSBoost Featrues1 2 folds")
disp.figure_.savefig("Test Confusion Matrix RUS Featrues1_2 folds.png")
print(f"Test Confusion matrix RUS:\n{disp.confusion_matrix}"'\n')
    
disp = metrics.plot_confusion_matrix(mlp, X_test, y_test)
disp.figure_.suptitle(f"Test Confusion Matrix MLP Featrues1 2 folds")
disp.figure_.savefig("Test Confusion Matrix MLP Featrues1_2 folds.png")
print(f"Test Confusion matrix MLP:\n{disp.confusion_matrix}"'\n')


#Cria a matriz de confusão de treino
disp = metrics.plot_confusion_matrix(xgb, X_train, y_train)
disp.figure_.suptitle(f"Training Confusion Matrix XGBoost Featrues1 2 folds")
disp.figure_.savefig("Training Confusion Matrix XGB Featrues1_2 folds.png")
print(f"Training Confusion matrix XGB:\n{disp.confusion_matrix}"'\n')
        
disp = metrics.plot_confusion_matrix(lgbm, X_train, y_train)
disp.figure_.suptitle(f"Training Confusion Matrix LGBM Featrues1 2 folds")
disp.figure_.savefig("Training Confusion Matrix LGBM Featrues1_2 folds.png")
print(f"Training Confusion matrix LGBM:\n{disp.confusion_matrix}"'\n')
        
disp = metrics.plot_confusion_matrix(rfc, X_train, y_train)
disp.figure_.suptitle(f"Training Confusion Matrix RFC Featrues1 2 folds")
disp.figure_.savefig("Training Confusion Matrix RFC Featrues1_2 folds.png")
print(f"Training Confusion matrix RFC:\n{disp.confusion_matrix}"'\n')
        
disp = metrics.plot_confusion_matrix(etc, X_train, y_train)
disp.figure_.suptitle(f"Training Confusion Matrix ETC Featrues1 2 folds")
disp.figure_.savefig("Training Confusion Matrix ETC Featrues1_2 folds.png")
print(f"Training Confusion matrix ETC:\n{disp.confusion_matrix}"'\n')

disp = metrics.plot_confusion_matrix(rus, X_train, y_train)
disp.figure_.suptitle(f"Training Confusion Matrix RUSBoost Featrues1 2 folds")
disp.figure_.savefig("Training Confusion Matrix RUS Featrues1_2 folds.png")
print(f"Training Confusion matrix RUS:\n{disp.confusion_matrix}"'\n')
    
disp = metrics.plot_confusion_matrix(mlp, X_train, y_train)
disp.figure_.suptitle(f"Training Confusion Matrix MLP Featrues1 2 folds")
disp.figure_.savefig("Training Confusion Matrix MLP Featrues1_2 folds.png")
print(f"Training Confusion matrix MLP:\n{disp.confusion_matrix}"'\n')

"""
#Cria a matriz de confusão de teste
disp = metrics.plot_confusion_matrix(xgb, X_test, y_test)
disp.figure_.suptitle(f"Test Confusion Matrix XGBoost Featrues2 2 folds")
disp.figure_.savefig("Test Confusion Matrix XGB Featrues2_2 folds.png")
print(f"Test Confusion Matrix XGB:\n{disp.confusion_matrix}"'\n')
        
disp = metrics.plot_confusion_matrix(lgbm, X_test, y_test)
disp.figure_.suptitle(f"Test Confusion Matrix LGBM Featrues2 2 folds")
disp.figure_.savefig("Test Confusion Matrix LGBM Featrues2_2 folds.png")
print(f"Test Confusion Matrix LGBM:\n{disp.confusion_matrix}"'\n')
        
disp = metrics.plot_confusion_matrix(rfc, X_test, y_test)
disp.figure_.suptitle(f"Test Confusion Matrix RFC Featrues2 2 folds")
disp.figure_.savefig("Test Confusion Matrix RFC Featrues2_2 folds.png")
print(f"Test Confusion matrix RFC:\n{disp.confusion_matrix}"'\n')
        
disp = metrics.plot_confusion_matrix(etc, X_test, y_test)
disp.figure_.suptitle(f"Test Confusion Matrix ETC Featrues2 2 folds")
disp.figure_.savefig("Test Confusion Matrix ETC Featrues2_2 folds.png")
print(f"Test Confusion matrix ETC:\n{disp.confusion_matrix}"'\n')

disp = metrics.plot_confusion_matrix(rus, X_test, y_test)
disp.figure_.suptitle(f"Test Confusion Matrix RUSBoost Featrues2 2 folds")
disp.figure_.savefig("Test Confusion Matrix RUS Featrues2_2 folds.png")
print(f"Test Confusion matrix RUS:\n{disp.confusion_matrix}"'\n')
    
disp = metrics.plot_confusion_matrix(mlp, X_test, y_test)
disp.figure_.suptitle(f"Test Confusion Matrix MLP Featrues2 2 folds")
disp.figure_.savefig("Test Confusion Matrix MLP Featrues2_2 folds.png")
print(f"Test Confusion matrix MLP:\n{disp.confusion_matrix}"'\n')


#Cria a matriz de confusão de treino
disp = metrics.plot_confusion_matrix(xgb, X_train, y_train)
disp.figure_.suptitle(f"Training Confusion Matrix XGBoost Featrues2 2 folds")
disp.figure_.savefig("Training Confusion Matrix XGB Featrues2_2 folds.png")
print(f"Training Confusion matrix XGB:\n{disp.confusion_matrix}"'\n')
        
disp = metrics.plot_confusion_matrix(lgbm, X_train, y_train)
disp.figure_.suptitle(f"Training Confusion Matrix LGBM Featrues2 2 folds")
disp.figure_.savefig("Training Confusion Matrix LGBM Featrues2_2 folds.png")
print(f"Training Confusion matrix LGBM:\n{disp.confusion_matrix}"'\n')
        
disp = metrics.plot_confusion_matrix(rfc, X_train, y_train)
disp.figure_.suptitle(f"Training Confusion Matrix RFC Featrues2 2 folds")
disp.figure_.savefig("Training Confusion Matrix RFC Featrues2_2 folds.png")
print(f"Training Confusion matrix RFC:\n{disp.confusion_matrix}"'\n')
        
disp = metrics.plot_confusion_matrix(etc, X_train, y_train)
disp.figure_.suptitle(f"Training Confusion Matrix ETC Featrues2 2 folds")
disp.figure_.savefig("Training Confusion Matrix ETC Featrues2_2 folds.png")
print(f"Training Confusion matrix ETC:\n{disp.confusion_matrix}"'\n')

disp = metrics.plot_confusion_matrix(rus, X_train, y_train)
disp.figure_.suptitle(f"Training Confusion Matrix RUSBoost Featrues2 2 folds")
disp.figure_.savefig("Training Confusion Matrix RUS Featrues2_2 folds.png")
print(f"Training Confusion matrix RUS:\n{disp.confusion_matrix}"'\n')
    
disp = metrics.plot_confusion_matrix(mlp, X_train, y_train)
disp.figure_.suptitle(f"Training Confusion Matrix MLP Featrues2 2 folds")
disp.figure_.savefig("Training Confusion Matrix MLP Featrues2_2 folds.png")
print(f"Training Confusion matrix MLP:\n{disp.confusion_matrix}"'\n')


#Cria a matriz de confusão de teste
disp = metrics.plot_confusion_matrix(xgb, X_test, y_test)
disp.figure_.suptitle(f"Test Confusion Matrix XGBoost Featrues3 2 folds")
disp.figure_.savefig("Test Confusion Matrix XGB Featrues3_2 folds.png")
print(f"Test Confusion Matrix XGB:\n{disp.confusion_matrix}"'\n')
        
disp = metrics.plot_confusion_matrix(lgbm, X_test, y_test)
disp.figure_.suptitle(f"Test Confusion Matrix LGBM Featrues3 2 folds")
disp.figure_.savefig("Test Confusion Matrix LGBM Featrues3_2 folds.png")
print(f"Test Confusion Matrix LGBM:\n{disp.confusion_matrix}"'\n')
        
disp = metrics.plot_confusion_matrix(rfc, X_test, y_test)
disp.figure_.suptitle(f"Test Confusion Matrix RFC Featrues3 2 folds")
disp.figure_.savefig("Test Confusion Matrix RFC Featrues3_2 folds.png")
print(f"Test Confusion matrix RFC:\n{disp.confusion_matrix}"'\n')
        
disp = metrics.plot_confusion_matrix(etc, X_test, y_test)
disp.figure_.suptitle(f"Test Confusion Matrix ETC Featrues3 2 folds")
disp.figure_.savefig("Test Confusion Matrix ETC Featrues3_2 folds.png")
print(f"Test Confusion matrix ETC:\n{disp.confusion_matrix}"'\n')

disp = metrics.plot_confusion_matrix(rus, X_test, y_test)
disp.figure_.suptitle(f"Test Confusion Matrix RUSBoost Featrues3 2 folds")
disp.figure_.savefig("Test Confusion Matrix RUS Featrues3_2 folds.png")
print(f"Test Confusion matrix RUS:\n{disp.confusion_matrix}"'\n')
    
disp = metrics.plot_confusion_matrix(mlp, X_test, y_test)
disp.figure_.suptitle(f"Test Confusion Matrix MLP Featrues3 2 folds")
disp.figure_.savefig("Test Confusion Matrix MLP Featrues3_2 folds.png")
print(f"Test Confusion matrix MLP:\n{disp.confusion_matrix}"'\n')


#Cria a matriz de confusão de treino
disp = metrics.plot_confusion_matrix(xgb, X_train, y_train)
disp.figure_.suptitle(f"Training Confusion Matrix XGBoost Featrues3 2 folds")
disp.figure_.savefig("Training Confusion Matrix XGB Featrues3_2 folds.png")
print(f"Training Confusion matrix XGB:\n{disp.confusion_matrix}"'\n')
        
disp = metrics.plot_confusion_matrix(lgbm, X_train, y_train)
disp.figure_.suptitle(f"Training Confusion Matrix LGBM Featrues3 2 folds")
disp.figure_.savefig("Training Confusion Matrix LGBM Featrues3_2 folds.png")
print(f"Training Confusion matrix LGBM:\n{disp.confusion_matrix}"'\n')
        
disp = metrics.plot_confusion_matrix(rfc, X_train, y_train)
disp.figure_.suptitle(f"Training Confusion Matrix RFC Featrues3 2 folds")
disp.figure_.savefig("Training Confusion Matrix RFC Featrues3_2 folds.png")
print(f"Training Confusion matrix RFC:\n{disp.confusion_matrix}"'\n')
        
disp = metrics.plot_confusion_matrix(etc, X_train, y_train)
disp.figure_.suptitle(f"Training Confusion Matrix ETC Featrues3 2 folds")
disp.figure_.savefig("Training Confusion Matrix ETC Featrues3_2 folds.png")
print(f"Training Confusion matrix ETC:\n{disp.confusion_matrix}"'\n')

disp = metrics.plot_confusion_matrix(rus, X_train, y_train)
disp.figure_.suptitle(f"Training Confusion Matrix RUSBoost Featrues3 2 folds")
disp.figure_.savefig("Training Confusion Matrix RUS Featrues3_2 folds.png")
print(f"Training Confusion matrix RUS:\n{disp.confusion_matrix}"'\n')
    
disp = metrics.plot_confusion_matrix(mlp, X_train, y_train)
disp.figure_.suptitle(f"Training Confusion Matrix MLP Featrues3 2 folds")
disp.figure_.savefig("Training Confusion Matrix MLP Featrues3_2 folds.png")
print(f"Training Confusion matrix MLP:\n{disp.confusion_matrix}"'\n')
"""

print(f"Classification report for classifier {xgb}:\n"
          f"{metrics.classification_report(y_test, pred_xgb)}\n")
print(f"Classification report for classifier {lgbm}:\n"
          f"{metrics.classification_report(y_test, pred_lgbm)}\n")
print(f"Classification report for classifier {rfc}:\n"
          f"{metrics.classification_report(y_test, pred_rfc)}\n")
print(f"Classification report for classifier {etc}:\n"
          f"{metrics.classification_report(y_test, pred_etc)}\n")
print(f"Classification report for classifier {rus}:\n"
          f"{metrics.classification_report(y_test, pred_rus)}\n")
print(f"Classification report for classifier {mlp}:\n"
          f"{metrics.classification_report(y_test, pred_mlp)}\n")
