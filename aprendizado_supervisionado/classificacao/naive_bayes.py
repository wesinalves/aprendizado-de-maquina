'''
Algoritmo Naive beyes usando Python

considerando a lei da probabilidade condicional

P(A|B) = [P(B|A)*P(A)] / P(B)

O classificador naive beyes é definido por:
y = arg max_i P(y_i|x)

by Wesin Alves.
'''

from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm
import pandas as pd
import numpy as np

def load_balance():
    '''Carrega dados para treinar o modelo.'''
    attributes = ['Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance']
    columns = ['Class']
    columns.extend(attributes)
    dataset = pd.read_csv('../../datasets/balance-scale.data', names=columns)
    data_X = dataset[attributes]
    data_y = dataset['Class']
    return data_X, data_y


def apriori(y):
    '''Calcular a probabilidade a priori das classes.'''
    Py_i = {}
    total = len(y)    
    for key, value in enumerate(y.value_counts()):
        Py_i[y.value_counts().index[key]] = value / total
    return Py_i
    

def train(X, y):
    '''Calcular média e desvio padrão de atributos dada a classe.'''
    train_data = X.copy()
    train_data['Class'] = y
    means = {}
    sigmas = {}    
    for x in X.columns:
        for c in y.value_counts().index:          
            filtered_x = train_data[train_data.Class == c][x]
            means[f'{x}|{c}'] = filtered_x.mean()
            sigmas[f'{x}|{c}'] = filtered_x.std()
    
    return means, sigmas           
            

def gerar_grafico():
    '''Gera gráfico do resultado.'''        
    
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)    
    ani = FuncAnimation(fig, update_plot, 
         frames=range(len(y_pred)), interval=50)

    plt.title('naive bayes algorithm')
    plt.xlabel('peso esquerdo')
    plt.ylabel('distancia esquerda')    
    ani.save('naivebayes.gif', writer='imagemagick')

def update_plot(i):
    scatter = ax.scatter(X_test.iloc[i,0], X_test.iloc[i,1])
    y_cc = pd.Categorical(y_pred)    
    scatter.set_facecolor(colors[y_cc.astype('category').codes[i]])

if __name__ == '__main__':
    # carregar base de dados
    X, y = load_balance()
    
    # inicializa variáveis
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        stratify=y, test_size=0.26, random_state=42)
    
    
    # calcular a probabilidade a priori da classe P(y_i)
    p_y = apriori(y_train)        
    
    # calcular a probabilidade condicional p(X|y_i)
    means, sigmas = train(X_train, y_train)
    
    # P(X|y_i) = P(x_1|y_i) * P(x_2|y_i) ... P(x_d|y_i)
    py_X = []
    for j in y_test.value_counts().index:
        pX_y = {}
        summation = np.zeros(X_test.shape[0])
        for i in X_test.columns:
            snd = norm(means[f'{i}|{j}'], sigmas[f'{i}|{j}'])
            pX_y[f'{i}|{j}'] = np.log10(snd.pdf(X_test[i]))
            summation += pX_y[f'{i}|{j}']
            
        py_X.append(np.log10(p_y[j]) + summation)
    
    y_pred = y_test.value_counts().index[np.argmax(py_X, axis=0)]    
    
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(cm)
    print(acc)
    
    fig, ax = plt.subplots()
    colors = ['blue', 'red', 'green']
    scatter = ax.clear()  
    gerar_grafico()

    
