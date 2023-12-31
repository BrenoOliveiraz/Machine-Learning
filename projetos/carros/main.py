import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

uri = 'https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv'

dados = pd.read_csv(uri)

# Mapeia a coluna 'sold' de 'yes' e 'no' para 1 e 0
torf = {
    'yes': 1,
    'no': 0
}
dados.sold = dados.sold.map(torf)

x = dados.drop(columns=["sold"], axis=1)
y = dados['sold']

# Calcula a idade dos modelos dos carros e converte a quilometragem por ano para quilômetros
currentYear = datetime.today().year
dados['model_age'] = currentYear - dados.model_year
dados["km_per_year"] = dados.mileage_per_year * 1.60934

# Removendo colunas desnecessarias
dados = dados.drop(columns=["Unnamed: 0", "mileage_per_year", "model_year"], axis=1)
print(dados.head())

# Define uma semente (seed) para reprodutibilidade dos resultados
SEED = 20
np.random.seed(SEED)

# Dados brutos para treinamento
raw_train_x, raw_test_x, train_y, test_y = train_test_split(x, y, stratify=y, test_size=0.25)

# Padronizando os dados passo a passo
scaler = StandardScaler()
scaler.fit(raw_train_x)
train_x = scaler.transform(raw_train_x)
test_x = scaler.transform(raw_test_x)

model = DecisionTreeClassifier(max_depth=3)
model.fit(raw_train_x, train_y)
previsoes = model.predict(raw_test_x)
acuracy = accuracy_score(test_y, previsoes) *100
print("A acuracia foi de %.2f%%" %acuracy)

from sklearn.tree import plot_tree
# Grafico
plt.figure(figsize=(20, 5))
plot_tree(model, filled=True, rounded=True, feature_names=x.columns.tolist(), class_names=["No", "Yes"])
plt.show()


