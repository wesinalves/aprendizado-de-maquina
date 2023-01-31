'''
Algoritmo 1R usando Python

escolhe a regra com a menor taxa de erros.

by Wesin Alves.
'''

from math import sqrt
from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

def load_balance():
    '''Carrega dados para treinar o modelo.'''
    attributes = ['Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance']
    columns = ['Class']
    columns.extend(attributes)
    dataset = pd.read_csv('../../datasets/balance-scale.data', names=columns)
    data_X = dataset[attributes]
    data_y = dataset['Class']
    return data_X, data_y

def train(X, y):
    '''Contabiliza as os erros de cada atributo por classe'''
    train_data = X.copy()
    train_data['Class'] = y

    error_rate = {}
    rules = {}
    for attribute in X.columns:        
        error_rules = 0
        for v in train_data[attribute].value_counts().index:
            # count how often each class appears per attribute
            classes_frequency = train_data[(train_data[attribute] == v)].Class.value_counts()
             
            # find the most frequent class
            most_frequent = classes_frequency.idxmax()
            
            # save the current rule
            rules[attribute] = rules.get(attribute, {})
            rules[attribute][v] = most_frequent
            
            # actual attribute and not most frequent class
            errors = train_data[
                (train_data[attribute] == v) & 
                ~(train_data['Class'] == most_frequent)
            ]
            
            rows = train_data[train_data[attribute] == v]
            error_rules += len(errors) / len(rows)
        
        error_rate[attribute] = error_rules
    
    # get attribute with low error
    best_attribute = min(error_rate)
            
    return rules[best_attribute], best_attribute
 
if __name__ == '__main__':
    # carregar base de dados
    X, y = load_balance()
    # inicializa variáveis
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        stratify=y, test_size=0.3, random_state=42)

    rule, best_attr = train(X_train, y_train)
    
    y_pred = []
    
    for index, row in X_test.iterrows():        
        value = row[best_attr]
        y_pred.append(rule[value])
    
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(cm)
    print(acc)    
