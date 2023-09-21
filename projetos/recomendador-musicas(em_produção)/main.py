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

# print(pca_pipeline[1].explained_variance_ratio_.sum()) #quantidade de explicação dentro do PCA em porcentagem 
# print(pca_pipeline[1].explained_variance_.sum()) #quantidade de explicação em variavel x e y

#CLUSTERIZAÇÃO POR MUSICA
from sklearn.preprocessing import OneHotEncoder #como se fosse um getdummies aplicado a pipeline
ohe = OneHotEncoder(dtype=int)
colum_ohe = ohe.fit_transform(general_data[['artists']]).toarray()
general_data_without_artist= general_data.drop('artists', axis=1)
music_dummie = pd.concat([general_data_without_artist, pd.DataFrame(colum_ohe, columns=ohe.get_feature_names_out(['artists']))], axis=1)
# print(music_dummie)

#tracando pipeline com padronização do standardScaler e reduzindo os grupos importantes com o PCA
pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=0.7, random_state=SEED))])
music_embedding_pca = pca_pipeline.fit_transform(music_dummie.drop(['id', 'name', 'artists_song'], axis=1)) 
projection_music = pd.DataFrame(data=music_embedding_pca) 
# print(projection_music)

#aplicação cluster com k-means
kmeans_pca_pipeline = KMeans(n_clusters=50, verbose=False, random_state=SEED)
kmeans_pca_pipeline.fit(projection_music)
general_data['cluster_pca'] = kmeans_pca_pipeline.predict(projection_music)
projection_music['cluster_pca'] = kmeans_pca_pipeline.predict(projection_music) #salvando no meu dataframe uma nova coluna com as previsoes 
projection_music['artists'] = general_data['artists']
projection_music['song'] = general_data['artists_song']
# print(projection_music)

#plotando projeções
# pic4 = px.scatter(projection_music, 0, 1, color='cluster_pca', hover_data=[0, 1, 'song'])
# pic4.show()

#RECOMENDAÇÕES DE MUSICA
from sklearn.metrics.pairwise import euclidean_distances
music_name = 'Ed Sheeran - Shape of You'
cluster = list(projection_music[projection_music['song'] == music_name]['cluster_pca'])[0] #compara a musica que eu setei com o predict de todas as musicas para identificar o numero do cluster
to_recommend = projection_music[projection_music['cluster_pca']==cluster][[0, 1, 'song']] #adiciona todas as musicas da cluster encontrada com base nas colunas 0, 1 e song
music_x= list(projection_music[projection_music['song']== music_name][0])[0] #localiza e grava os valores da coluna x (0)
music_y= list(projection_music[projection_music['song']== music_name][1])[0] #localiza e grava os valores da coluna y (1) *as duas primeiraos colunas explicam melhor o predict
print(music_x)
#calcular a distancias euclidianas entre as musicas mais proximas do cluster
distances= euclidean_distances(to_recommend[[0, 1]], [[music_x, music_y]])


to_recommend['id'] = general_data['id'] #adicionando o id de volta, para eventual href, caso deseje
to_recommend['distances'] = distances #adicionando, tambem, a distancia ou proximidade de uma musica com as demais, para escalonar uma recomendação
recommendations = to_recommend.sort_values('distances').head(10)
print(recommendations)





