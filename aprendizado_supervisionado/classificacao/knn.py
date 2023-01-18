'''
Algoritmo Knn usando Python

considerando a distância euclidiana,

seleciona a classe majoritária dos K vizinhos mais próximos.

by Wesin Alves.
'''

from math import sqrt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import itertools

def distancia_euclidiana(vet_a, vet_b):
    '''Calcula distância euclidiana entre dois vetores.'''
    
    if len(vet_a) != len(vet_b):
        raise('Os vetores precisam ter o mesmo comprimento.')
    
    somatorio = 0
    for a,b in zip(vet_a, vet_b):
        somatorio += (a - b)**2
    
    return sqrt(somatorio)

def selecionar_kvizinhos(x, vizinhos, k):
    '''Seleciona os K vizinhos mais próximos de x.'''
    
    distancias = {}
    for index, v in enumerate(vizinhos):
        dist = distancia_euclidiana(x, v)
        distancias[index] = dist
    
    distancias = dict(sorted(distancias.items(), key=lambda x:x[1]))
    k_vizinhos = list(distancias.keys())[:k]    
    
    return k_vizinhos


def votar_classe(k_vizinhos, alvos, classes):
    '''Seleciona a classe mais votada.'''
    
    votos = [0 for _ in classes]
    for k, _ in enumerate(k_vizinhos):
        for c, classe in enumerate(classes):
            if classe == alvos[k]:
                votos[c] = votos[c] + 1

    return votos

def selecionar_mais_votada(votos):
    '''Seleciona classe mais votada'''
    mais_votada = 0
    for i, v in enumerate(votos):        
        if v > votos[mais_votada]:
            mais_votada = i

    return mais_votada

def gerar_grafico():
    '''Gera gráfico do resultado.'''        
    
    ax.set_xlim(4, 8)
    ax.set_ylim(2, 5)    
    ani = FuncAnimation(fig, update_plot, 
         frames=range(len(y_pred)), interval=50)

    plt.title('Knn algorithm')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')    
    ani.save('knn.gif', writer='imagemagick')

def update_plot(i):    
    scatter = ax.scatter(X_test[i,0], X_test[i,1]) 
    scatter.set_facecolor(colors[y_pred[i]]) 

if __name__ == '__main__':
    # carregar base de dados
    X, y = load_iris(return_X_y=True)
    # inicializa variáveis
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        stratify=y, test_size=0.3, random_state=42)
    # seleciona valor do parâmetro K
    K = 9

    # para cada exemplo de treinamento, calcular distância euclidiana
    # entre o novo ponto e o ponto de treinamento i.
    y_pred = []
    for x in X_test:
        k_vizinhos = selecionar_kvizinhos(x, X_train, K)
        
        # seleciona os K pontos mais próximos
        # do novo ponto
        classes = [0, 1, 2]
        k_alvos = [y_test[k] for k in k_vizinhos]
        votos = votar_classe(
            k_vizinhos,
            k_alvos,
            classes,        
            )

        # verifica a classe mais votada
        mais_votada = selecionar_mais_votada(votos)

        # atribui a classe majoritária ao novo ponto
        y_pred.append(classes[mais_votada])        
    
    
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(cm)
    print(acc)
    
    fig, ax = plt.subplots()
    colors = ['blue', 'red', 'green']
    scatter = ax.clear()  
    gerar_grafico()

    
