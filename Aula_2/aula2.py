# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 10:45:38 2022

@author: ernan
"""

import numpy as np

"""
my_arr = np.arange(1000000)
my_list = list(range(1000000))
%time for _ in range(10): my_arr2 = my_arr * 2
%time for _ in range(10): my_list2 = [x * 2 for x in my_list]
"""

"""
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d[2])
print(arr2d[:2,1:])
print(arr2d[::-1])
"""
"""
import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)

import matplotlib.pyplot as plt
plt.plot(walk[:100])
plt.show()


nsteps = 10
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
"""


"""
Exercícios
1)) Escreva um programa NumPy que cria um array de duas dimensões com 1’s nas bordas e 0’s no centro:
[[ 1. 1. 1. 1. 1.]
[ 1. 0. 0. 0. 1.]
[ 1. 0. 0. 0. 1.]
[ 1. 0. 0. 0. 1.]
[ 1. 1. 1. 1. 1.]]
2)) Escreva um programa NumPy que cria uma borda de 0’s em volta de um array de duas dimensões (consulte o
método np.pad):
[[ 0. 0. 0. 0. 0.]
[ 0. 1. 1. 1. 0.]
[ 0. 1. 1. 1. 0.]
[ 0. 1. 1. 1. 0.]
[ 0. 0. 0. 0. 0.]]
3)) Escreva um programa NumPy que converte uma array com valores de temperaturas em Graus Celcius para
Graus Farenheit.
$`C = (5*(F-32))/9`$
4)) Escreva um programa NumPy que cria um array bidimensional de 0’s e 1’s gerando um padrão de “tabuleiro de
xadrez”. Por exemplo:
[[0 1 0 1 0 1 0 1]
[1 0 1 0 1 0 1 0]
[0 1 0 1 0 1 0 1]
[1 0 1 0 1 0 1 0]
[0 1 0 1 0 1 0 1]
[1 0 1 0 1 0 1 0]
[0 1 0 1 0 1 0 1]
[1 0 1 0 1 0 1 0]]


##Exercicio 1
matriz1 = np.ones((5,5))
##matriz1[1:matriz1.shape[0]-1,1:matriz1.shape[1]-1] = 0 
matriz1[1:-1,1:-1] = 0

matriz1b = np.empty((5,5))
matriz1b[:] = 1 
matriz1b[1:matriz1.shape[0]-1,1:matriz1.shape[1]-1] = 0 

##Exercicio 2
matriz0 = np.zeros((5,5))
matriz0[1:matriz0.shape[0]-1,1:matriz0.shape[1]-1] = 1 

matriz0P = np.ones((5,5))
matriz0P = np.pad(matriz0P, pad_width=1,mode='constant',constant_values=0)

##Exercicio 3
## F = ((9*C)/5)+32
celcius = np.array([20,30,40])

farenheit = ((celcius*9)/5) + 32

##Exercicio 4
matrizX = np.zeros((8,8))
matrizX[1:matrizX.shape[0]:2,0:matrizX.shape[1]:2] = 1
matrizX[0:matrizX.shape[0]:2,1:matrizX.shape[1]:2] = 1

matrizXT = np.tile(np.array([[0,1],[1,0]]),(4,4))

"""

import pandas as pd

obj = pd.Series([4, 7, -5, 3])
obj2 = pd.Series([4, 7, -6, 3],index=['a','b','c','d'])
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = pd.Series(sdata)

states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = pd.Series(sdata, index=states)

obj5 = obj3 + obj4


data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
'year': [2000, 2001, 2002, 2001, 2002, 2003],
'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)

newdata = {'state': 'Ohio',
'year': 2004,
'pop': 3.9}

data = pd.DataFrame(np.arange(16).reshape((4, 4)),
index=['Ohio', 'Colorado', 'Utah', 'New York'],
columns=['one', 'two', 'three', 'four'])


