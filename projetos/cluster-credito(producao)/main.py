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
data_norm = Normalizer().fit_transform(data.values)

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
    return s, dbs, calinski

s1, dbs1, calinski1 = clustering_algorithm(5, data_norm)


# Silhouete quantifica a qualidade de um único ponto de dados em relação ao cluster a que pertence e aos clusters vizinhos.
# varia de -1 a 1, onde um valor maior indica que o objeto está bem enquadrado em seu próprio cluster e mal enquadrado em clusters vizinhos. 


# Davis bouldin baseada na distancia do ponto pro centroid(ponto central do cluster, a média de todos os pontos do cluster para cada um dos atributos)
# Mede a "similaridade" entre cada cluster e seu cluster mais próximo vizinho. Quanto menor o valor do DB, melhor é a qualidade da clusterização.


# Índice de Calinski mede a razão entre a dispersão média entre clusters e a dispersão média dentro dos clusters. Ele avalia o quão bem os clusters estão separados e definidos.
# Quanto maior o valor do índice CH, melhor é a qualidade da clusterização


#testando nossa validação com um conjunto de dados aleatorio
random_data = np.random.rand(8950,16)
s2, dbs2, calinski2 = clustering_algorithm(5, random_data)
print(f"Silhouete {s2: .2f}, Davis Bouldin {dbs2: .2f}, Calinski Harabasz {calinski2: .2f}")
print(f"Silhouete {s1: .2f}, Davis Bouldin {dbs1: .2f}, Calinski Harabasz {calinski1: .2f}")

# Validando a estabilidade dos dados
# Dividir a base de dados e rodar o KMeans para cada uma dessas divisões para tesstar a estabilidade dos dados
set1, set2, set3 = np.array_split(data_norm, 3)
s1, dbs1, calinski1 = clustering_algorithm(5, set1)
s2, dbs2, calinski2 = clustering_algorithm(5, set2)
s3, dbs3, calinski3 = clustering_algorithm(5, set3)
print(f"{s1: .2f}, {dbs1: .2f}, {calinski1: .2f}")
print(f"{s2: .2f}, {dbs2: .2f}, {calinski2: .2f}")
print(f"{s3: .2f}, {dbs3: .2f}, {calinski3: .2f}")




