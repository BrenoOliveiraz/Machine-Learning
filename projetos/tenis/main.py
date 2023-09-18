import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Definindo a semente para reprodutibilidade
SEED = 20
np.random.seed(SEED)

# URL do conjunto de dados
uri = 'https://raw.githubusercontent.com/MathMachado/DSWP/master/Dataframes/tennis.csv'

# Carregando os dados do CSV
dados = pd.read_csv(uri)

swap = {
    'No': 0,
    'Yes': 1
}

swap2 ={
    "D1": 1, "D2": 2, 'D3': 3, 'D4': 4, "D5": 5, "D6": 6, "D7": 7, "D8": 8, "D9": 9, "D10": 10, "D11": 11, "D12":12, "D13":13, "D14":14, "D15":15
}

dados['play'] = dados['play'].replace(swap)
dados['day'] = dados['day'].replace(swap2)

# Selecionar as colunas categóricas para transformar em variáveis dummy
colunas_categoricas = ["outlook", "temp", "humidity", "wind"]

# Aplicar get_dummies às colunas categóricas
dummie_dados = pd.get_dummies(dados[colunas_categoricas], drop_first=True)

# Concatenar as colunas dummy com o DataFrame original
dados = pd.concat([dados, dummie_dados], axis=1)

# Remover as colunas originais que foram convertidas em dummies
dados.drop(colunas_categoricas, axis=1, inplace=True)

print(dados.head())

x = dados.drop(columns=["play"], axis=1)
y = dados["play"]

train_x, test_x, train_y, test_y = train_test_split(x, y, stratify=y , random_state=SEED, test_size=0.25)

model = DecisionTreeClassifier(max_depth=3)
model.fit(train_x, train_y)

previsoes = model.predict(test_x)

acuracy = accuracy_score(test_y, previsoes) *100
print("A acuracia foi de %.2f%%" %acuracy)

from sklearn.tree import plot_tree

# Configurar o tamanho da figura
plt.figure(figsize=(10, 8))

# Plotar a árvore de decisão
plot_tree(model, filled=True, rounded=True, feature_names=x.columns.tolist(), class_names=["No", "Yes"])


# Mostrar o gráfico
plt.show()



"""


model = LinearSVC()
model.fit(train_x, train_y)

previsoes = model.predict(test_x)

acuracy = accuracy_score(test_y, previsoes) *100
print("A acuracia foi de %.2f%%" %acuracy)
"""