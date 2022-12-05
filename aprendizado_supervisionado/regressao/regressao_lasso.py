'''
Regressão lasso usando Python

considerando y = X * beta,

y: vetor com n elementos
X: matriz com n,m elementos
beta: vetor de n coeficientes

Forma matricial para calcular o coeficiente beta considerando a 
penalização lasso. 

Considerando termo de penalização l1 = alpha * sum(abs(beta))
Como derivada do termo de penalização em relação 
ao coeficiente é alpha * sign(beta),
onde n é o número de elementos em beta


beta = (X.T @ X)^-1 @ X.T @ y - alpha * sign(beta)
'''

from sklearn.metrics import mean_squared_error as rmse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def regressao_lasso(X, y, alpha):
    '''Implementa a regressão ridge.'''

    # encontrar o vetor beta    
    pseudo_inverse = np.linalg.inv(X.T @ X) @ X.T
    beta = (pseudo_inverse @ y)
    beta_lasso = beta - alpha * np.sign(beta)

    return beta_lasso

def gerar_dados():
    '''Gerar matriz X e vetor y.'''

    rng = np.random.default_rng()
    X = 5 * rng.random((100, 3))
    X[:, 0] = 1
    beta = 2 * rng.random((3, 1))
    y = np.dot(X, beta) + 0.7

    return X, y

def main():
    '''Executa a função principal.'''

    X, y = gerar_dados()
    alpha = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    # estimar valores de y
    b = regressao_lasso(X_train, y_train, alpha)
    y_pred = np.dot(X_test, b)

    # avaliar o modelo
    erro = rmse(y_test, y_pred)
    print(erro)

    # plotar gráficos
    fig, axis = plt.subplots(X.shape[1])
    fig.suptitle('Regressão lasso em Python')
    for i,axs in enumerate(axis):
        axs.scatter(X_test[:,i], y_test, label='Valor real')
        axs.scatter(X_test[:,i], y_pred, label='Valor estimado')
        axs.set(ylabel=f'Dimensão {i}')
    
    plt.legend()
    plt.xlabel('Valores de X')
    plt.show()

main()



