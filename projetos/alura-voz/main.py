# Importando a biblioteca pandas para trabalhar com dados tabulares
import pandas as pd
import numpy as np
# Carregando o conjunto de dados a partir de uma URL
dados = pd.read_csv('https://raw.githubusercontent.com/alura-cursos/ML_Classificacao_por_tras_dos_panos/main/Dados/Customer-Churn.csv')

# Modificando manualmente algumas colunas usando um dicionário de tradução
traducao_dic = {'Sim': 1, 'Nao': 0}
dadosmodificados = dados[['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn']].replace(traducao_dic)

# Transformando as variáveis categóricas em variáveis dummy
dummie_dados = pd.get_dummies(dados.drop(['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn'], axis=1))

# Combinando as variáveis transformadas com as originais
dados_final = pd.concat([dadosmodificados, dummie_dados], axis=1)

# Exemplo de uma entrada (dados de Maria)
Xmaria = [[0, 0, 1, 1, 0, 0, 39.90, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]]

# Divisão em inputs (X) e outputs (y)
X = dados_final.drop('Churn', axis=1)
y = dados_final['Churn']

# Importando a biblioteca para padronizar os dados
from sklearn.preprocessing import StandardScaler
norm = StandardScaler()

# Padronizando os dados
X_normalizado = norm.fit_transform(X)
Xmaria_normalizado = norm.transform(pd.DataFrame(Xmaria, columns=X.columns))

# Calculando a distância Euclidiana entre o exemplo de Maria e o primeiro exemplo nos dados padronizados

a = Xmaria_normalizado
b = X_normalizado[0]
distancia = np.sqrt(np.sum(np.square(a - b)))

# Dividindo os dados em conjuntos de treinamento e teste
from sklearn.model_selection import train_test_split
X_treino, X_teste, y_treino, y_teste = train_test_split(X_normalizado, y, test_size=0.3, random_state=123)

# Importando a biblioteca para criar o modelo KNN (K-Nearest Neighbors)
from sklearn.neighbors import KNeighborsClassifier
# Instanciando o modelo KNN com a métrica de distância euclidiana
knn = KNeighborsClassifier(metric='euclidean', n_neighbors=10)

# Treinando o modelo com os dados de treinamento
knn.fit(X_treino, y_treino)

from sklearn.metrics import accuracy_score
# Realizando previsões de churn
predito_knn = knn.predict(X_teste)
acuracy = accuracy_score(y_teste, predito_knn) *100
print("A acuracia foi de %.2f%%" %acuracy)