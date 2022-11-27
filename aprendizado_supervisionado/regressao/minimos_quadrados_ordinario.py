'''
Algoritmo mínimos quadrados ordinário usando Python

considerando y_i = m*x_i + b,

onde m é o coeficiente angular,
e b é o coeficiente angular

Y = vetor com n elementos y_n
X = vetor com n elementos x_n

m = n(sum(X*Y) - sum(X)*sum(Y)) / 
    (n * sum(X**2) - sum(X)**2) 

b = (sum(Y) - m*sum(X)) / n

'''

from sklearn.metrics import mean_squared_error as rmse
import matplotlib.pyplot as plt
from random import uniform

def minimos_quadrados(X, Y):
    '''Implementa método mínimos quadrados.'''

    # encontrar o coeficiente angular m
    n = len(X)
    sumxy = sum([x * y for x,y in zip(X, Y)])
    sumx = sum(X)
    sumy = sum(Y)
    sumx2 = sum([x * x for x in X])

    m = (n * sumxy - sumx * sumy) / (n * sumx2 - sumx**2)

    # encontrar o coeficiente linear b
    b = (sumy - m * sumx) / n

    return m,b

def gerar_vetores():
    '''Gerar vetores de X e Y.'''
    m, b = 2, 5
    X = [x for x in range(10)]
    Y = [m * x + b + uniform(0, 2) for x in X]

    return X, Y

def main():
    '''Executa função principal.'''

    X, Y = gerar_vetores()

    # estimar valores de Y
    m, b = minimos_quadrados(X, Y)
    Y_pred = [m * x + b for x in X]

    # avaliar modelo
    erro = rmse(Y, Y_pred)
    print(erro)
    print(m, b)

    # plotar gráficos
    gtrue = plt.scatter(X, Y)
    gpred = plt.scatter(X, Y_pred)
    plt.legend((gtrue, gpred), ('Y Real', 'Y Estimado'))
    plt.xlabel('Valores de X')
    plt.ylabel('Valores de Y')
    plt.show()

main()


