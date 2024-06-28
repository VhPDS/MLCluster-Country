# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 19:57:57 2024

@author: torug
"""

#%% Instalando os pacotes

!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install plotly
!pip install scipy
!pip install scikit-learn
!pip install pingouin


#%% Importando os pacotes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.stats as stats
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import pingouin as pg
import plotly.express as px 
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'

#%%

dados_paises = pd.read_csv('dados_paises.csv')
tabela_descritiva = dados_paises.describe()

paises = dados_paises.drop(columns='country')
#Padronizando a diferença entre as váriaveis utilizando o z-score

paises_pad = paises.apply(zscore, ddof=1)

dist_euclidiana = pdist(paises_pad, metric='euclidean')

plt.figure(figsize=(16,8))
dend_single = sch.linkage(paises_pad, method='single', metric='euclidean')
single = sch.dendrogram(dend_single)
plt.title('country Single Linkage')
plt.ylabel('country')
plt.xlabel('euclidean distance')
plt.show()

plt.figure(figsize=(16,8))
dend_average = sch.linkage(paises_pad, method='average', metric='euclidean')
average = sch.dendrogram(dend_average)
plt.title('country Single Linkage')
plt.ylabel('country')
plt.xlabel('euclidean distance')
plt.show()



plt.figure(figsize=(16,8))
dend_complete= sch.linkage(paises_pad, method='complete', metric='euclidean')
complete = sch.dendrogram(dend_complete)
plt.title('country Complete Linkage')
plt.ylabel('country')
plt.xlabel('euclidean distance')
plt.show()

#%%

#Adicionando as indicações de cluster ao dataset

cluster_complete = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='complete')
indica_cluster_complete = cluster_complete.fit_predict(paises_pad)
dados_paises['cluster_complete'] = indica_cluster_complete
paises_pad['cluster_complete'] = indica_cluster_complete
dados_paises['cluster_complete'] = dados_paises['cluster_complete'].astype('category')
paises_pad['cluster_complete'] = paises_pad['cluster_complete'].astype('category')


#%%

#Analise de variância de um fator

# child_mort
pg.anova(dv='child_mort', 
         between='cluster_complete', 
         data=paises_pad,
         detailed=True).T

# exports
pg.anova(dv='exports', 
         between='cluster_complete', 
         data=paises_pad,
         detailed=True).T

# imports
pg.anova(dv='imports', 
         between='cluster_complete', 
         data=paises_pad,
         detailed=True).T

# health
pg.anova(dv='health', 
         between='cluster_complete', 
         data=paises_pad,
         detailed=True).T

# income
pg.anova(dv='income', 
         between='cluster_complete', 
         data=paises_pad,
         detailed=True).T

# inflation
pg.anova(dv='inflation', 
         between='cluster_complete', 
         data=paises_pad,
         detailed=True).T

# life_expec
pg.anova(dv='life_expec', 
         between='cluster_complete', 
         data=paises_pad,
         detailed=True).T

# total_fer
pg.anova(dv='total_fer', 
         between='cluster_complete', 
         data=paises_pad,
         detailed=True).T

# gdpp
pg.anova(dv='gdpp', 
         between='cluster_complete', 
         data=paises_pad,
         detailed=True).T
#%%

#Gerando gráfico

fig = px.scatter_3d(dados_paises,
                    x='total_fer',
                    y='income',
                    z='life_expec',
                    color='cluster_complete')
fig.show()

#%%

#Identificando as caracteristicas do cluster

analise_paises = dados_paises.drop(columns=['country']).groupby(by=['cluster_complete'])
tab_medias = analise_paises.mean().T
tab_desc = analise_paises.describe().T