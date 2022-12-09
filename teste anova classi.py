# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 10:56:49 2022

@author: nayan
"""

from statsmodels.stats.multicomp import plot_simultaneous
import seaborn as sns
import pandas as pd
import scipy.stats as stats

# obter tabela ANOVA como R como saída
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ANOVA table using bioinfokit v1.0.3 or later (it uses wrapper script for anova_lm)
from bioinfokit.analys import stat

"""
# load data 
dados = pd.read_excel(f'Resultados_Total_CARAC.xlsx', sheet_name=0) #ACC
dados1 = pd.read_excel(f'Resultados_Total_CARAC.xlsx', sheet_name=1) #PRECISÃO
dados2 = pd.read_excel(f'Resultados_Total_CARAC.xlsx', sheet_name=2) #RECALL
dados3 = pd.read_excel(f'Resultados_Total_CARAC.xlsx', sheet_name=3) #F1SCORE

# reshape the d dataframe suitable for statsmodels package 
dados_melt = pd.melt(dados.reset_index(), id_vars=['index'], value_vars=['XGB',	'LGBM',	'RFC',	'ETC',	'RUS',	'MLP'])
dados1_melt = pd.melt(dados1.reset_index(), id_vars=['index'], value_vars=['XGB',	'LGBM',	'RFC',	'ETC',	'RUS',	'MLP'])
dados2_melt = pd.melt(dados2.reset_index(), id_vars=['index'], value_vars=['XGB',	'LGBM',	'RFC',	'ETC',	'RUS',	'MLP'])
dados3_melt = pd.melt(dados3.reset_index(), id_vars=['index'], value_vars=['XGB',	'LGBM',	'RFC',	'ETC',	'RUS',	'MLP'])
# replace column names
dados_melt.columns = ['index', 'Classificadores', 'ACC']
dados1_melt.columns = ['index', 'Classificadores', 'Precisão']
dados2_melt.columns = ['index', 'Classificadores', 'Recall']
dados3_melt.columns = ['index', 'Classificadores', 'F1Score']


#TESTE ANOVA ACC
# funções estatísticas f_oneway tomam os grupos como entrada e devolve valor ANOVA F e p
fvalue, pvalue = stats.f_oneway(dados['XGB'], dados['LGBM'], dados['RFC'], dados['ETC'],
                                dados['RUS'], dados['MLP'])
print('\n Os valores f e p para ACC são:\n', fvalue, pvalue)

# realizamos o teste do Mínimo Quadrado Ordinário (OLS)
model = ols('ACC ~ C(Classificadores)', data=dados_melt).fit()
#Tabela Anova para um ou mais modelos lineares ajustados
anova_table = sm.stats.anova_lm(model, typ=2)
print('\n Os valores obtidos a partir da análise ANOVA é:\n', anova_table)

res = stat()
res.anova_stat(df=dados_melt, res_var='ACC', anova_model='ACC ~ C(Classificadores)')
print('\n',res.anova_summary)

#TUCKEY
res0 = stat()
res0.tukey_hsd(df=dados_melt, res_var='ACC', xfac_var='Classificadores',
               anova_model='ACC ~ C(Classificadores)')
print('\n', res0.tukey_summary)


#TESTE ANOVA PRECISÃO
# funções estatísticas f_oneway tomam os grupos como entrada e devolve valor ANOVA F e p
fvalue1, pvalue1 = stats.f_oneway(dados1['XGB'], dados1['LGBM'], dados1['RFC'], dados1['ETC'],
                                dados1['RUS'], dados1['MLP'])
print('\n Os valores f e p para Precisão são:\n', fvalue1, pvalue1)

# realizamos o teste do Mínimo Quadrado Ordinário (OLS) 
model1 = ols('Precisão ~ C(Classificadores)', data=dados1_melt).fit()
#Tabela Anova para um ou mais modelos lineares ajustados
anova_table1 = sm.stats.anova_lm(model1, typ=2)
print('\n Os valores obtidos a partir da análise ANOVA é:\n', anova_table1)

res1 = stat()
res1.anova_stat(df=dados1_melt, res_var='Precisão', anova_model='Precisão ~ C(Classificadores)')
print('\n',res1.anova_summary)

#TUCKEY
res1A = stat()
res1A.tukey_hsd(df=dados1_melt, res_var='Precisão', xfac_var='Classificadores', anova_model='Precisão ~ C(Classificadores)')
print('\n', res1A.tukey_summary)

#TESTE ANOVA RECALL
# funções estatísticas f_oneway tomam os grupos como entrada e devolve valor ANOVA F e p
fvalue2, pvalue2 = stats.f_oneway(dados2['XGB'], dados2['LGBM'], dados2['RFC'], dados2['ETC'],
                                dados2['RUS'], dados2['MLP'])
print('\n Os valores f e p para Recall são:\n', fvalue2, pvalue2)

# realizamos o teste do Mínimo Quadrado Ordinário (OLS)
model2 = ols('Recall ~ C(Classificadores)', data=dados2_melt).fit()
#Tabela Anova para um ou mais modelos lineares ajustados
anova_table2 = sm.stats.anova_lm(model2, typ=2)
print('\n Os valores obtidos a partir da análise ANOVA é:\n', anova_table2)

res2 = stat()
res2.anova_stat(df=dados2_melt, res_var='Recall', anova_model='Recall ~ C(Classificadores)')
print('\n',res2.anova_summary)

#TUCKEY
res2A = stat()
res2A.tukey_hsd(df=dados2_melt, res_var='Recall', xfac_var='Classificadores', anova_model='Recall ~ C(Classificadores)')
print('\n', res2A.tukey_summary)

#TESTE ANOVA F1SCORE
# funções estatísticas f_oneway tomam os grupos como entrada e devolve valor ANOVA F e p
fvalue3, pvalue3 = stats.f_oneway(dados3['XGB'], dados3['LGBM'], dados3['RFC'], dados3['ETC'],
                                dados3['RUS'], dados3['MLP'])
print('\n Os valores f e p para F1Score são:\n', fvalue3, pvalue3)

# realizamos o teste do Mínimo Quadrado Ordinário (OLS)
model3 = ols('F1Score ~ C(Classificadores)', data=dados3_melt).fit()
#Tabela Anova para um ou mais modelos lineares ajustados
anova_table3 = sm.stats.anova_lm(model3, typ=2)
print('\n Os valores obtidos a partir da análise ANOVA é:\n', anova_table3)

res3 = stat()
res3.anova_stat(df=dados3_melt, res_var='F1Score', anova_model='F1Score ~ C(Classificadores)')
print('\n',res3.anova_summary)

#TUCKEY
res3A = stat()
res3A.tukey_hsd(df=dados3_melt, res_var='F1Score', xfac_var='Classificadores', anova_model='F1Score ~ C(Classificadores)')
print('\n', res3A.tukey_summary)


dataframe= pd.DataFrame(res0.tukey_summary)
dataframe1= pd.DataFrame(res1A.tukey_summary)
dataframe2= pd.DataFrame(res2A.tukey_summary)
dataframe3= pd.DataFrame(res3A.tukey_summary)


with pd.ExcelWriter('testeC_TUCKEY_ACC.xlsx') as writer:
    dataframe.to_excel(writer)

with pd.ExcelWriter('testeC_TUCKEY_P.xlsx') as writer:
    dataframe1.to_excel(writer)
    
with pd.ExcelWriter('testeC_TUCKEY_R.xlsx') as writer:
    dataframe2.to_excel(writer)

with pd.ExcelWriter('testeC_TUCKEY_F1_.xlsx') as writer:
    dataframe3.to_excel(writer)
    
"""

# load data 
dados = pd.read_excel(f'Resultados_Total_CARAC.xlsx', sheet_name=4) #ACC
dados1 = pd.read_excel(f'Resultados_Total_CARAC.xlsx', sheet_name=5) #PRECISÃO
dados2 = pd.read_excel(f'Resultados_Total_CARAC.xlsx', sheet_name=6) #RECALL
dados3 = pd.read_excel(f'Resultados_Total_CARAC.xlsx', sheet_name=7) #F1SCORE

# reshape the d dataframe suitable for statsmodels package 
dados_melt = pd.melt(dados.reset_index(), id_vars=['index'], value_vars=['XGB',	'LGBM',	'RFC',	'ETC',	'RUS',	'MLP'])
dados1_melt = pd.melt(dados1.reset_index(), id_vars=['index'], value_vars=['XGB',	'LGBM',	'RFC',	'ETC',	'RUS',	'MLP'])
dados2_melt = pd.melt(dados2.reset_index(), id_vars=['index'], value_vars=['XGB',	'LGBM',	'RFC',	'ETC',	'RUS',	'MLP'])
dados3_melt = pd.melt(dados3.reset_index(), id_vars=['index'], value_vars=['XGB',	'LGBM',	'RFC',	'ETC',	'RUS',	'MLP'])
# replace column names
dados_melt.columns = ['index', 'Classificadores', 'ACC']
dados1_melt.columns = ['index', 'Classificadores', 'Precisão']
dados2_melt.columns = ['index', 'Classificadores', 'Recall']
dados3_melt.columns = ['index', 'Classificadores', 'F1Score']



#TESTE ANOVA ACC
# funções estatísticas f_oneway tomam os grupos como entrada e devolve valor ANOVA F e p
fvalue, pvalue = stats.f_oneway(dados['XGB'], dados['LGBM'], dados['RFC'], dados['ETC'],
                                dados['RUS'], dados['MLP'])
print('\n Os valores f e p para ACC são:\n', fvalue, pvalue)

# realizamos o teste do Mínimo Quadrado Ordinário (OLS)
model = ols('ACC ~ C(Classificadores)', data=dados_melt).fit()
#Tabela Anova para um ou mais modelos lineares ajustados
anova_table = sm.stats.anova_lm(model, typ=2)
print('\n Os valores obtidos a partir da análise ANOVA é:\n', anova_table)

res = stat()
res.anova_stat(df=dados_melt, res_var='ACC', anova_model='ACC ~ C(Classificadores)')
print('\n',res.anova_summary)

#TUCKEY
res0 = stat()
res0.tukey_hsd(df=dados_melt, res_var='ACC', xfac_var='Classificadores',
               anova_model='ACC ~ C(Classificadores)')
print('\n', res0.tukey_summary)


#TESTE ANOVA PRECISÃO
# funções estatísticas f_oneway tomam os grupos como entrada e devolve valor ANOVA F e p
fvalue1, pvalue1 = stats.f_oneway(dados1['XGB'], dados1['LGBM'], dados1['RFC'], dados1['ETC'],
                                dados1['RUS'], dados1['MLP'])
print('\n Os valores f e p para Precisão são:\n', fvalue1, pvalue1)

# realizamos o teste do Mínimo Quadrado Ordinário (OLS) 
model1 = ols('Precisão ~ C(Classificadores)', data=dados1_melt).fit()
#Tabela Anova para um ou mais modelos lineares ajustados
anova_table1 = sm.stats.anova_lm(model1, typ=2)
print('\n Os valores obtidos a partir da análise ANOVA é:\n', anova_table1)

res1 = stat()
res1.anova_stat(df=dados1_melt, res_var='Precisão', anova_model='Precisão ~ C(Classificadores)')
print('\n',res1.anova_summary)

#TUCKEY
res1A = stat()
res1A.tukey_hsd(df=dados1_melt, res_var='Precisão', xfac_var='Classificadores', anova_model='Precisão ~ C(Classificadores)')
print('\n', res1A.tukey_summary)

#TESTE ANOVA RECALL
# funções estatísticas f_oneway tomam os grupos como entrada e devolve valor ANOVA F e p
fvalue2, pvalue2 = stats.f_oneway(dados2['XGB'], dados2['LGBM'], dados2['RFC'], dados2['ETC'],
                                dados2['RUS'], dados2['MLP'])
print('\n Os valores f e p para Recall são:\n', fvalue2, pvalue2)

# realizamos o teste do Mínimo Quadrado Ordinário (OLS)
model2 = ols('Recall ~ C(Classificadores)', data=dados2_melt).fit()
#Tabela Anova para um ou mais modelos lineares ajustados
anova_table2 = sm.stats.anova_lm(model2, typ=2)
print('\n Os valores obtidos a partir da análise ANOVA é:\n', anova_table2)

res2 = stat()
res2.anova_stat(df=dados2_melt, res_var='Recall', anova_model='Recall ~ C(Classificadores)')
print('\n',res2.anova_summary)

#TUCKEY
res2A = stat()
res2A.tukey_hsd(df=dados2_melt, res_var='Recall', xfac_var='Classificadores', anova_model='Recall ~ C(Classificadores)')
print('\n', res2A.tukey_summary)

#TESTE ANOVA F1SCORE
# funções estatísticas f_oneway tomam os grupos como entrada e devolve valor ANOVA F e p
fvalue3, pvalue3 = stats.f_oneway(dados3['XGB'], dados3['LGBM'], dados3['RFC'], dados3['ETC'],
                                dados3['RUS'], dados3['MLP'])
print('\n Os valores f e p para F1Score são:\n', fvalue3, pvalue3)

# realizamos o teste do Mínimo Quadrado Ordinário (OLS)
model3 = ols('F1Score ~ C(Classificadores)', data=dados3_melt).fit()
#Tabela Anova para um ou mais modelos lineares ajustados
anova_table3 = sm.stats.anova_lm(model3, typ=2)
print('\n Os valores obtidos a partir da análise ANOVA é:\n', anova_table3)

res3 = stat()
res3.anova_stat(df=dados3_melt, res_var='F1Score', anova_model='F1Score ~ C(Classificadores)')
print('\n',res3.anova_summary)

#TUCKEY
res3A = stat()
res3A.tukey_hsd(df=dados3_melt, res_var='F1Score', xfac_var='Classificadores', anova_model='F1Score ~ C(Classificadores)')
print('\n', res3A.tukey_summary)



"""
dataframe= pd.DataFrame(res0.tukey_summary)
dataframe1= pd.DataFrame(res1A.tukey_summary)
dataframe2= pd.DataFrame(res2A.tukey_summary)
dataframe3= pd.DataFrame(res3A.tukey_summary)


with pd.ExcelWriter('testeC_TUCKEY_ACC5.xlsx') as writer:
    dataframe.to_excel(writer)

with pd.ExcelWriter('testeC_TUCKEY_P5.xlsx') as writer:
    dataframe1.to_excel(writer)
    
with pd.ExcelWriter('testeC_TUCKEY_R5.xlsx') as writer:
    dataframe2.to_excel(writer)

with pd.ExcelWriter('testeC_TUCKEY_F1_5.xlsx') as writer:
    dataframe3.to_excel(writer)
    
"""