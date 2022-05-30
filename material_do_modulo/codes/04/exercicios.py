import numpy as np

# Escreva um programa NumPy que cria um array de duas dimensões com 1’s nas 
# bordas e 0’s no centro:
m = np.ones((10,10))
m[1:-1,1:-1] = 0

# Escreva um programa NumPy que cria uma borda de 0’s em volta de um array de 
# duas dimensões (consulte o método np.pad):
m = np.ones((5,5))
m = np.pad(m, pad_width=1, mode='constant', constant_values=0)

# Escreva um programa NumPy que converte um array com valores de temperaturas 
# em Graus Celcius para Graus Farenheit.
# C = (5*(F-32))/9
# F = ((C*9)/5)-32
arr_f = np.array([0, 10, 20, 30, 40, 200])
arr_c = (5*(arr_f-32))/9
arr2_f = ((arr_c*9)/5)+32

# Escreva um programa NumPy que cria um array bidimensional de 0’s e 1’s 
# gerando um padrão de “tabuleiro de xadrez”:
m = np.tile( np.array([0,1]), (4,4))

