'''
Algoritmo Árvore de decisão usando Python

### A árvore de decisão é um algorítmo softisticado que utiliza a 
estratégia dividir para conquistar e é formada por: 

* nó folhas
* nó de divisão

### Algorítmo

CriaArvore(T)
- Selecionar atributo que maximiza critério de divisão 
em relação as classes
- Criar uma subárvore para atributo
- para cada subárvore s
-- se cada exemplo em s for mesma classe ? cria um nó folha com a classe:
    aplicar mesmo procedimento recursivamente para cada 
    subconjunto de treinamento
    CriaArvore(T)

- retornar a árvore de decisão

### critério de divisão

- entropia E(p) = - sum(pi * log2(pi)) -> expressa a aleatoriedade de p
- ganho de informação  G(d) = E(p) - E(s,d), 
onde d é o candidato a divisão da árvore e 
E(s, d) = sum(di * E(di))
- gini i(p) = 1 - sum(pi²)

### Algoritmo para calcular o ganho de informação
1. Calcular a entropia do conjunto de treino T
2. Para cada atributo T, dividir em subconjuntos Ti
(i) calcular entropia de cada subconjunto ti E(Ti)
(ii) calcular a entropia ponderada E(T, ti)
(iii) calcular o ganho de informação G(T, ti)

-- ganho da informação para atributos categóricos (entropia ponderada)
-- ganho da informação para atributos númericos (discretizar)

Dado um conjunto de exemplos classificados, qual atributo selecionar

by Wesin Alves.
'''