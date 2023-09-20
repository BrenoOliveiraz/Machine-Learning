import pandas as pd
import numpy as np
SEED=1234
np.random.seed(SEED)
pd.set_option('display.max_columns', None)

uri = 'projetos/recomendador-musicas(em_produção)/Dados_totais.csv'
uri2 = 'projetos/recomendador-musicas(em_produção)/data_by_genres.csv'
uri3 = 'projetos/recomendador-musicas(em_produção)/data_by_year.csv'
general_data = pd.read_csv(uri)
data_gender = pd.read_csv(uri2)
data_years = pd.read_csv(uri3)

##eliminando colunas que não entrarão no modelo
# print(data['year'].unique())
# print(data.shape)
general_data = general_data.drop(['explicit', 'key', 'mode'], axis=1)
# print(f'DADOS GERAIS \n{data.head}()')
# print(data.shape)
# print(data.isnull().sum())
# print(data.isna().sum())


data_gender = data_gender.drop(['key', 'mode'], axis=1)
# print(data_years['year'].unique())

data_years= data_years[data_years['year']>=2000]
data_years = data_years.drop(['key', 'mode'], axis=1)
# print(data_years['year'].unique())
data_years = data_years.reset_index() #resetando o index depois do tratamento ">=2000", ele iniciou de 79
# print(f" DADOS POR ANOS  \n{data_years.head()}")

import plotly.express as px
pic1 = px.line(data_years, x='year', y='loudness', markers=True, title='Variação do Loudness conforme os anos')
# pic1.show()

import plotly.graph_objects as go
pic2 = go.Figure()
pic2.add_trace(go.Scatter(x=data_years['year'], y=data_years['acousticness'], name='Acousticness'))
pic2.add_trace(go.Scatter(x=data_years['year'], y=data_years['valence'], name='Valence'))
pic2.add_trace(go.Scatter(x=data_years['year'], y=data_years['danceability'], name='Danceability'))
pic2.add_trace(go.Scatter(x=data_years['year'], y=data_years['energy'], name='Energy'))
pic2.add_trace(go.Scatter(x=data_years['year'], y=data_years['instrumentalness'], name='Acousticness'))
pic2.add_trace(go.Scatter(x=data_years['year'], y=data_years['liveness'], name='Liveness'))
pic2.add_trace(go.Scatter(x=data_years['year'], y=data_years['speechiness'], name='Speechiness'))
# pic2.show()

#CLUSTERIZAÇÃO POR GENERO

without_row_gender = data_gender.drop('genres', axis=1)
# print(without_row_gender.head(2))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA #uma serie de analises matematicas feitas para reduzir as milhares de colunas em uma quantidade menor, com base no valor 'n' q atribuimos

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2, random_state=SEED))])

genre_embedding_pca = pca_pipeline.fit_transform(without_row_gender) 
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding_pca) 

#utilizando o k-means para identificar os pontos de dados semelhantes e agrupa-los
from sklearn.cluster import KMeans
kmeans_pca = KMeans(n_clusters=5, verbose=False, random_state=SEED)
kmeans_pca.fit(projection)
data_gender['cluster_pca'] = kmeans_pca.predict(projection)
projection['cluster_pca'] = kmeans_pca.predict(projection) #salvando no meu dataframe de x,y
projection['genres'] = data_gender['genres']
# print(projection)

#plotando
pic3 = px.scatter(projection, 'x', 'y', color='cluster_pca', hover_data=['x', 'y', 'genres'])
# pic3.show()

print(pca_pipeline[1].explained_variance_ratio_.sum()) #quantidade de explicação dentro do PCA em porcentagem 
print(pca_pipeline[1].explained_variance_.sum()) #quantidade de explicação em variavel x e y










