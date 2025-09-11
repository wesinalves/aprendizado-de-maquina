'''
Regressão logística usando sklearn/python

Regressão logística é um classificador binário usado para estimar a probabilidade de uma instância pertencer a uma classe particular. Se a probabilidade estimada for maior que 50%, então o modelo prediz que a instância pertence a classe positiva, caso contrário, pertence a classe negativa.

Referência:
2019, Géron. Hands on machine learning with scikit learning, keras and tensorflow


by Wesin Alves
'''

from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris["data"][:, 3:] # petal width
print(X[:10])
y = (iris["target"]==2).astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)


log_reg = LogisticRegression(penalty='l2', solver='sag', max_iter=20)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("----------Matriz de confusão -------------")
print(cm)
print("-----------------------------------------")
print(f"Acurácia do modelo: {acc}")


X_new = np.linspace(0, 3, 1000).reshape(-1,1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:,1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:,0], "b--", label="Not Iris-Virginica")
plt.ylabel("Probability")
plt.xlabel("Petal width")
plt.legend()
plt.show()
