# -*- coding: utf-8 -*-
"""Construa seu primeiro modelo de aprendizado de máquina em python.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fpv65cLkwEy49Yvn_vGRUv73tMjGZTJY

# Meu primeiro projeto de Machine Learning
1. Carregar conjunto de dados
2. Análise exploratória dos dados
3. Pré-processamento dos dados
  * 3.1 Separar atributos e alvo
  * 3.2 Dividir dados de treino e teste
4. Criar modelo de Regresssão Linear
5. Criar modelo de Árvore de Decisão
6. Comparar modelos
7. Visualizar dados

# Carregar conjunto de dados
"""

import pandas as pd

df = pd.read_csv("FuelConsumptionRatings2023.csv", encoding="latin1")
df.head()

df.shape

df.describe()

df.info()

"""# Pré-processamento dos dados"""

df.isnull().sum()

# manipulando valores em falta
df.dropna(inplace=True)

# removendo atributos irrelevantes
df = df.drop(['Transmission', 'Make', 'Year', 'Vehicle Class', 'Model'], axis=1)

# converter atributo categórico (texto para número)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
categorical_columns = ['Fuel Type']

for column in categorical_columns:
  df[column] = label_encoder.fit_transform(df[column])

df.head()

"""## Separar atributos (X) de alvo (y)"""

X = df.drop(['CO2 Emissions (g/km)'], axis=1)
y = df['CO2 Emissions (g/km)']

"""## Dividir dados de treino e teste"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""# Regressão linear

# Criando o modelo
"""

from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print ('Coefficients: ', model.coef_)

y_pred_lr_train = lr_model.predict(X_train)
y_pred_lr_test = lr_model.predict(X_test)
y_pred_lr_test

"""## Avaliando o desempenho do modelo"""

from sklearn.metrics import mean_squared_error, r2_score

lr_train_mse = mean_squared_error(y_train, y_pred_lr_train)
lr_train_r2 = r2_score(y_train, y_pred_lr_train)

lr_test_mse = mean_squared_error(y_test, y_pred_lr_test)
lr_test_r2 = r2_score(y_test, y_pred_lr_test)

"""### O coeficiente de determinação, também chamado de R², é uma medida de ajuste de um modelo estatístico linear generalizado, como a regressão linear simples ou múltipla, aos valores observados de uma variável aleatória. O R² varia entre 0 e 1, por vezes sendo expresso em termos percentuais. Nesse caso, expressa a quantidade da variância dos dados que é explicada pelo modelo linear. Assim, quanto maior o R², mais explicativo é o modelo linear, ou seja, melhor ele se ajusta à amostra. Por exemplo, um R² = 0,8234 significa que o modelo linear explica 82,34% da variância da variável dependente a partir do regressores (variáveis independentes) incluídas naquele modelo linear."""

print(f"LR MSE train error", lr_train_mse)
print(f"LR R2 train error", lr_train_r2)
print(f"LR MSE test error", lr_test_mse)
print(f"LR R2 test error", lr_test_r2)

lr_results = pd.DataFrame(["Linear Regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ["Method", "Training MSE", "Training R2", "Test MSE", "Test R2"]

lr_results

"""# Árvore de decisão

## Criando o modelo
"""

from sklearn import tree
from matplotlib import pyplot as plt
tree_model = tree.DecisionTreeRegressor()
tree_model.fit(X_train, y_train)
tree.plot_tree(tree_model, proportion=True, max_depth=2, fontsize=7)
plt.show()

"""## Fazendo predições"""

y_pred_tree_train = tree_model.predict(X_train)
y_pred_tree_test = tree_model.predict(X_test)
y_pred_tree_test

"""## Avaliando o modelo"""

tree_train_mse = mean_squared_error(y_train, y_pred_tree_train)
tree_train_r2 = r2_score(y_train, y_pred_tree_train)

tree_test_mse = mean_squared_error(y_test, y_pred_tree_test)
tree_test_r2 = r2_score(y_test, y_pred_tree_test)

tree_results = pd.DataFrame(["AD Regression", tree_train_mse, tree_train_r2, tree_test_mse, tree_test_r2]).transpose()
tree_results.columns = ["Method", "Training MSE", "Training R2", "Test MSE", "Test R2"]

tree_results

"""
# Comparando Modelos"""

df_models = pd.concat([lr_results, tree_results])
df_models

"""# Visualizar dados do modelo"""

import numpy as np

plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred_lr_test, color='b', alpha=0.3)

z = np.polyfit(y_test, y_pred_lr_test, 1)
p = np.poly1d(z)

plt.plot(y_test, p(y_test), '#F8766D')
plt.ylabel("Predicted Values")
plt.xlabel("Acutal Values")
plt.title("Linear Regression Model")

plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred_tree_test, color='b', alpha=0.3)

z = np.polyfit(y_test, y_pred_tree_test, 1)
p = np.poly1d(z)

plt.plot(y_test, p(y_test), '#F8766D')
plt.ylabel("Predicted Values")
plt.xlabel("Acutal Values")
plt.title("Decision Tree Model")