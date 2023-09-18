# Importe as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Define a URI do conjunto de dados
uri = 'https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv'

# Carrega os dados do CSV
dados = pd.read_csv(uri)

# Mapeia a coluna 'sold' de 'yes' e 'no' para 1 e 0
torf = {
    'yes': 1,
    'no': 0
}
dados.sold = dados.sold.map(torf)

# Separa as features (variáveis independentes) e o target (variável dependente)
x = dados.drop(columns=["sold"], axis=1)
y = dados['sold']

# Calcula a idade dos modelos dos carros e converte a quilometragem por ano para quilômetros
currentYear = datetime.today().year
dados['model_age'] = currentYear - dados.model_year
dados["km_per_year"] = dados.mileage_per_year * 1.60934

# Remove colunas não utilizadas
dados = dados.drop(columns=["Unnamed: 0", "mileage_per_year", "model_year"], axis=1)

# Exibe as primeiras linhas dos dados
print(dados.head())

# Define uma semente (seed) para reprodutibilidade dos resultados
SEED = 20
np.random.seed(SEED)

# Divide os dados em conjuntos de treinamento e teste
raw_train_x, raw_test_x, train_y, test_y = train_test_split(x, y, stratify=y, test_size=0.25)

# Padroniza os dados usando StandardScaler
scaler = StandardScaler()
scaler.fit(raw_train_x)
train_x = scaler.transform(raw_train_x)
test_x = scaler.transform(raw_test_x)

model = DecisionTreeClassifier(max_depth=3)
model.fit(train_x, train_y)
previsoes = model.predict(test_x)
acuracy = accuracy_score(test_y, previsoes) *100
print("A acuracia foi de %.2f%%" %acuracy)

from sklearn.tree import plot_tree

# Configurar o tamanho da figura
plt.figure(figsize=(10, 5))

# Plotar a árvore de decisão
plot_tree(model, filled=True, rounded=True, feature_names=x.columns.tolist(), class_names=["No", "Yes"])


# Mostrar o gráfico
plt.show()


