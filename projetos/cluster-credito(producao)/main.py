import pandas as pd
import numpy as np
SEED=1234
np.random.seed(SEED)
pd.set_option('display.max_columns', None)

uri = 'projetos/cluster-credito/CC GENERAL.csv'

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
silhouete = metrics.silhouette_score(data_norm, labels, metric='euclidean')
print(silhouete)