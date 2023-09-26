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
distancia = np.sqrt(np.sum(np.square(a - b))) #distancia da maria para o client 0 da base

# Dividindo os dados em conjuntos de treinamento e teste
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X_normalizado, y, test_size=0.3, random_state=123)

# Utilizando modelo KNN (K-Nearest Neighbors)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(metric='euclidean', n_neighbors=10)
knn.fit(train_x, train_y)
predito_knn = knn.predict(test_x)

# Utilizando metodo probabilistico(Teorema de Naive Bayes)
from sklearn.naive_bayes import BernoulliNB
np.median(train_x) #tirar a media para colocar como parametro binario do modelo
bnb = BernoulliNB(binarize=0.44)
bnb.fit(train_x, train_y)
predito_bnb = bnb.predict(test_x)


#Utilizando metodo de arvore de decisões
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy', random_state=42)
dtc.fit(train_x, train_y)
predito_dtc= dtc.predict(test_x)


from sklearn.metrics import accuracy_score
acuracy = accuracy_score(test_y, predito_knn) *100
acuracy2 = accuracy_score(test_y, predito_bnb) *100
acuracy3 = accuracy_score(test_y, predito_dtc) *100
print(f"A taxa de acerto foi de KNN{acuracy: .2f}%, BNB{acuracy2: .2f}% e DTC{acuracy3: .2f}% ")



