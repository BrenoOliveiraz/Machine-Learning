import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Definindo a semente para reprodutibilidade
SEED = 20
np.random.seed(SEED)

# URL do conjunto de dados
uri = 'https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv'

# Carregando os dados do CSV
dados = pd.read_csv(uri)

# Trocando os valores de 1 (true) como "unfinished" para 0, para ser mais harmonioso
swap = {
    1: 0,
    0: 1
}

# Mapeando os valores da coluna 'unfinished' para 'finish'
dados['finish'] = dados.unfinished.map(swap)

# Dividindo os dados em treinamento e teste
x = dados[["expected_hours", "price"]]
y = dados["finish"]

raw_train_x, raw_test_x, train_y, test_y = train_test_split(x, y, stratify=y, test_size=0.25)

# Criação e treinamento do modelo SVC
model = SVC()
model.fit(raw_train_x, train_y)

# Realizando previsões
previsoes = model.predict(raw_test_x)

# Estabelecendo um algoritmo base para calcular a acurácia
baseline_previews = np.ones(540)

# Calculando a acurácia do algoritmo baseline
acuracy = accuracy_score(test_y, baseline_previews) * 100
print(f"A acurácia do algoritmo de baseline foi {acuracy:.2f}%")

# Criando um gráfico de dispersão usando Seaborn
sns.scatterplot(x="expected_hours", y="price", hue="finish", data=dados)

# Exibindo o gráfico na tela usando Matplotlib
plt.show()

# Usando o StandardScaler para criar uma escala redefinida entre dois valores específicos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(raw_train_x)
train_x = scaler.transform(raw_train_x)
test_x = scaler.transform(raw_test_x)

# Criando um novo modelo SVC com os dados padronizados e treinando o modelo
model = SVC()
model.fit(train_x, train_y)

# Realizando previsões com os dados padronizados
previsoes = model.predict(test_x)

# Calculando a acurácia do algoritmo após a padronização
acuracy = accuracy_score(test_y, previsoes) * 100
print(f"A acurácia do algoritmo após a padronização foi {acuracy:.2f}%")
plt.show()

# Definindo os valores mínimos e máximos para os eixos x e y
data_x = test_x[:, 0]
data_y = test_x[:, 1]

x_min = data_x.min()
x_max = data_x.max()
y_min = data_y.min()
y_max = data_y.max()

# Definindo a resolução da grade
pixels = 100

x_axis = np.arange(x_min, x_max, (x_max - x_min) / pixels)
y_axis = np.arange(y_min, y_max, (y_max - y_min) / pixels)
xx, yy = np.meshgrid(x_axis, y_axis)
points = np.c_[xx.ravel(), yy.ravel()]

# Realizando previsões para todos os pontos na grade
Z = model.predict(points)
Z = Z.reshape(xx.shape)

# Plotando o gráfico com Matplotlib
plt.scatter(data_x, data_y, c=test_y, s=1)
plt.contourf(xx, yy, Z, alpha=0.1)
plt.show()
