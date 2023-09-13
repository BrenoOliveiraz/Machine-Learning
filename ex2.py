import pandas as pd

uri='https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv'

dados = pd.read_csv(uri)

#usa o print(dados.head()) pra ver a leitura do csv e, assim, separar train_x e train_y

x = dados[["home", "how_it_works","contact"]] #usa-se dois colchetes pq esta extraindo uma lista py normal
y = dados["bought"] #quando extrair só uma coluna, não precisa colocar o colchete extra

#caso queira renomear as features de ingles pra portugues, é possivel utilizar uma funcionalidade do pandas, chamada "rename" atribuindo a um objeto, exemplo: dados.rename(columns = mapa) mapa = {..}

train_x = x[:75]
train_y = y[:75]
test_x = x[75:]
test_y = y[75:]

from sklearn.svm import LinearSVC #funcionalidade de juntar os dados com as labels
from sklearn.metrics import accuracy_score #funcionalidade de medir a precisão de acertos do predict
from sklearn.model_selection import train_test_split #importa a funcionalidade de separar o treino e o teste de um conj. de dados qualquer
model = LinearSVC()
model.fit(train_x, train_y)


previsoes = model.predict(test_x)

acuracy = accuracy_score(test_y, previsoes) *100
print("A acuracia foi de %.2f%%" %acuracy)

SEED = 20 #define um numero inicial para os algoritmos de geração aleatorios, para darem o mesmo resultado

#esse metodo serve pra substituir a divisão manual do treino com o teste, passando os parametros desejado dentro da função train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, stratify=y , #stratify serve para separar proporcionalmente de acordo com o y
                                                    random_state=SEED, test_size=0.25)

model = LinearSVC()
model.fit(train_x, train_y)

previsoes = model.predict(test_x)

acuracy = accuracy_score(test_y, previsoes) *100
print("A acuracia foi de %.2f%%" %acuracy)







