# features (1 sim, 0 não)
# pelo longo?
# perna curta?
# late?

#declarando as features
porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

# 1 => porco, 0 => cachorro
train_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3] #dados
train_y = [1,1,1,0,0,0] #label dos dados exemplo porco1: dado, 1(porco): label


#importar a library para
from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(train_x, train_y) #o fit é método responsável por juntar os dados com as labels

animal_misterioso = [1,1,1] #um exemplo aleatório criado para testar o código
model.predict([animal_misterioso])  #predict é a função da library que vai combinar o mistério com o fit realizado anteriormente

#gerando vários exemplos aleatórios para testar.
misterio1 = [1,1,1]
misterio2 = [1,1,0]
misterio3 = [0,1,1]

test_x = [misterio1, misterio2, misterio3]
test_y = [0, 1, 1]
previsoes = model.predict(test_x)




#se der print(previsoes == test_y) na comparação, o numpy vai comparar o predict com a atribuição do meu test_y e retornar um array dizendo se é true ou false

#para tirar a média de acertos
corretos = (previsoes == test_y).sum() #dando um print em corretos, ele retornara "[2]" q é o numero de match entre as previsões e a minha atribuição do test_y
total = len(test_x) #o numero total de elementos 3 (misterio1, misterio2, misterio3)
taxa_acertos = corretos/total * 100 #formula para tirar a media
print(taxa_acertos)

#importar nas metrics do sklearn a library accuracy para calcular a média

from sklearn.metrics import accuracy_score

taxa_de_acertos = accuracy_score(test_y, previsoes) #primeiro os verdadeiro, depois as previsoes

print(taxa_de_acertos)

