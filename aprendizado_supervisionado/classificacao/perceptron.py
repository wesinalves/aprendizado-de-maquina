'''
Perceptron usando apenas Python

Passos:
1. Inicializar
2. Calcular ativação u = w.T*x
3. Calcular saída y = signum(u) 
4. atualizar pesos w(n+1) = w(n) + lr*(d(n) - y)*x(n) 

by Wesin Alves.
'''

from math import sqrt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def limiar(value):
    val = value[0]
    if val > 0:
        val = 1.0
    else:
        val = -1.0
    
    return val
    

def quantizer(vector, threshold=152):
	x = []
	for val in vector:
		if val > threshold:
			x.append(1.0)
		else:
			x.append(-1.0)
	
	return np.array(x)

def gerar_grafico():
    '''Gera gráfico do resultado.'''        
    
      
    ani = FuncAnimation(fig, update_plot, 
         frames=range(len(y_pred)), interval=50)

    plt.title('Gráfico de erro')
    plt.xlabel('época')
    plt.ylabel('erro')    
    #ani.save('knn.gif', writer='imagemagick')
    plt.show()

def update_plot(i):    
    line.set_data(i, erros[i])
    return line,

if __name__ == '__main__':
    # carregar base de dados
    X, y = load_diabetes(return_X_y=True)
    # inicializa variáveis
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.7, random_state=42)

    epocas = 100      

    # inicializa e transforma as variáveis
    lr = 0.05
    b = np.ones((X_train.shape[0], 1))
    X_train = np.append(X_train, b, axis=1)
    w_n = np.zeros((X_train.shape[1], 1))    
    d_n = quantizer(y_train)
    
    mse = []

    # considerando o número de épocas de treinamento
    for i in range(epocas):
        erros = []
        y_pred = []
        for i, x in enumerate(X_train):

            # calcular o saída yj
            u = w_n.T @ x
            y_n = limiar(u)
            y_pred.append(y_n)
            
            
            # atualizar os pesos
            delta_w = lr * (d_n[i] - y_n) * x
            w_n = w_n + delta_w.reshape((delta_w.shape[0],1))
            
            # calcular erro
            erros.append(d_n[i] - y_n)
    
        mse.append(np.mean(np.array(erros)**2))

    print('MSE: ', mse)
    
    cm = confusion_matrix(d_n, y_pred)
    acc = accuracy_score(d_n, y_pred)
    print(cm)
    print(acc)    
     
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=3)
    plt.plot(mse)
    plt.title('Gráfico do erro de treinamento')
    plt.xlabel('época')
    plt.ylabel('erro')    
    plt.show()

    
