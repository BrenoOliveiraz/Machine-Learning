import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
uri = 'https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv'
dados = pd.read_csv(uri)
torf={
    'yes':1,
'no': 0
}
dados.sold = dados.sold.map(torf)
x = dados.drop(columns = ["sold"], axis=1 )
y= dados['sold']

from datetime import datetime
currentYear = datetime.today().year
dados['model_age'] = currentYear -dados.model_year
dados["km_per_yar"] = dados.mileage_per_year * 1.60934
dados = dados.drop(columns = ["Unnamed: 0", "mileage_per_year", "model_year"], axis = 1)
print(dados.head())

SEED = 20
np.random.seed(SEED)
raw_train_x, raw_test_x, train_y, test_y = train_test_split(x, y, stratify=y, test_size=0.25)

scaler = StandardScaler()
scaler.fit(raw_train_x)
train_x = scaler.transform(raw_train_x)
test_x = scaler.transform(raw_test_x)

#acuracy padrao svc
model = SVC()
model.fit(train_x, train_y)
previsoes = model.predict(test_x)
acuracy = accuracy_score(test_y, previsoes) * 100

print(f'a acuracia foi de {acuracy:.2f}')
