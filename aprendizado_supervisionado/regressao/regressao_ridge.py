'''
Regressão ridge usando Python

considerando y = X * beta,

Y = vetor com n elementos
X = matriz com n,m elementos

beta = (X.T * X + alpha * I)^-1 * X.T * y
'''

from sklearn.metrics import mean_squared_error as rmse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def regressao_ridge(X, y, alpha):
    '''Implementa regressão ridge.'''

    # encontrar o vetor beta
    A = alpha * np.eye(X.shape[1])
    pseudo_inverse = np.linalg.inv(X.T @ X + A) @ X.T
    beta = pseudo_inverse @ y 
    
    return beta

def gerar_dados():
    '''Gerar matriz X e vetor y.'''
    rng = np.random.default_rng()
    X = 5 * rng.random((100, 3))
    beta = 2 * rng.random((3, 1))
    y = np.dot(X, beta) + 0.7

    return X, y

def main():
    '''Executa função principal.'''

    X, y = gerar_dados()
    alpha = 1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    # estimar valores de Y
    b = regressao_ridge(X_train, y_train, alpha)
    y_pred = np.dot(X_test, b)  

    # avaliar modelo
    erro = rmse(y_test, y_pred)
    print(erro)    

    # plotar gráficos
    fig, axis = plt.subplots(X.shape[1])
    fig.suptitle('Regressão ridge em python')
    for i, axs in enumerate(axis):
        axs.scatter(X_test[:,i], y_test, label='Valor real')
        axs.scatter(X_test[:,i], y_pred, label='Valor estimado')
        axs.set(ylabel=f'Dimensão {i}')
    
    plt.legend()
    plt.xlabel('Valores de X')    
    plt.show()    

main()


