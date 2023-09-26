import pandas as pd
import numpy as np
SEED=1234
np.random.seed(SEED)
pd.set_option('display.max_columns', None)

uri = 'projetos/cluster-credito(producao)/CC GENERAL.csv'

data = pd.read_csv(uri)
data.drop(['CUST_ID', 'TENURE'], axis=1, inplace=True)

data.fillna(data.median(), inplace=True) #completa os dados que faltam no dataframe com a média dos demais
# print(data.isna().sum())

from sklearn.preprocessing import Normalizer
data_norm= Normalizer().fit_transform(data.values)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, n_init=10, max_iter=300)
y_pred = kmeans.fit_predict(data_norm)


from sklearn import metrics
labels = kmeans.labels_

def clustering_algorithm(n_clusters, data): #validação relativa
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(data)
    s = metrics.silhouette_score(data, labels, metric='euclidean')
    dbs = metrics.davies_bouldin_score(data, labels)
    calinski = metrics.calinski_harabasz_score(data, labels)
    return f" Silhouete {s}, Davis Bouldin {dbs}, Calinski Harabasz {calinski}"

print(clustering_algorithm(5, data_norm))

#Silhouete quantifica a qualidade de um único ponto de dados em relação ao cluster a que pertence e aos clusters vizinhos.
#varia de -1 a 1, onde um valor maior indica que o objeto está bem enquadrado em seu próprio cluster e mal enquadrado em clusters vizinhos. 


#matematica davis bouldin baseada na distancia do ponto pro centroid(ponto central do cluster, a média de todos os pontos do cluster para cada um dos atributos)
#mede a "similaridade" entre cada cluster e seu cluster mais próximo vizinho. Quanto menor o valor do DB, melhor é a qualidade da clusterização.


#Índice de Calinski mede a razão entre a dispersão média entre clusters e a dispersão média dentro dos clusters. Ele avalia o quão bem os clusters estão separados e definidos.
#Quanto maior o valor do índice CH, melhor é a qualidade da clusterização



