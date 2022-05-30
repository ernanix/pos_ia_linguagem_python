# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
"""
for i in [1, 2, 3, 4, 5]:

    # notice the blank line
    print(i)
"""    
"""
1. Escreva uma função que converte horas, minutos e segundos em um número total de segundos.

2. Escreva uma função que reconhece palíndromos.

3. Escreva uma função soma_de_quadrados(xs) que recebe uma lista de números xs 
e retorna a soma dos quadrados dos números na lista. Por exemplo 
soma_dos_quadrados([2, 3, 4]) deve retorna 4+9+16 que é 29.

4. Escreva uma função recursiva que calcula o fatorial de um número.

5. Escreva uma função recursiva para inverter uma lista.
"""

"""
def converte_segundos(h,m,s):
    return (h*60*60 + m*60 + s)

print(converte_segundos(1,20,45))


def palindromo(texto):
    texto2 = texto[::-1]
    return texto == texto2

print(palindromo("pdoap"))

def soma_de_quadrados(lista):
    lista2 = [x*2 for x in lista]
    return sum(lista2)    
    
print(soma_de_quadrados([3,4,2]))

def fatorial(num):
    if num > 0:
        return num * fatorial(num-1)
    else:
        return 1

print (fatorial(4))

def inverte_lista(lista,novalista = [],index = -1):
    try :
        novalista.append(lista[index])
        return inverte_lista(lista,novalista,index = index -1)
    except:
        return novalista
    
print(inverte_lista([1,2,3,4]))

"""
"""
pairs = [(x, y)
         for x in range(10)
         for y in range(10)] # 100 pairs (0,0) (0,1) ... (9,8), (9,9)
"""

assert 1 + 1 == 2
assert 1 + 1 == 2, "1 + 1 should equal 2 but didn't"

# exemplo com função
def smallest_item(xs):
    return min(xs)

assert smallest_item([10, 20, 5, 40]) == 5
assert smallest_item([1, 0, -1, 2]) == -1



"""
1.Utilize List Comprehension para elevar ao quadrado todos os elementos de uma lista;
2.Utilize List Comprehension para retirar elementos palíndromos de uma lista de strings;
3.Utilize List Comprehension para, dadas as listas A e B, criar uma lista C composta 
apenas pelos elementos presentes em ambas A e B;
"""

"""1"""

listasqr = [2,4,3,8]

sqr = [x*x for x in listasqr]

"""2"""
lista_strings = ["oco","ovo","ceu","mar"]

palin = [x for x in lista_strings if x == x[::-1]]
not_palin = [x for x in lista_strings if x != x[::-1]]

"""3"""
listaA = [1,2,3,4]
listaB = [4,5,6]
"""A e B"""
listaFimAnd = [x 
            for y in listaA
            for x in listaB
            if y == x]

listaFimAnd2 = [x for x in set(listaA+listaB) if x in listaA and x in listaB]

"""A ou B"""
listaFimOr = [x for x in set(listaA+listaB) if (x in listaA and x not in listaB) 
              or (x in listaB and x not in listaA) ]

listaFimOr2 = [*[y for y in listaA],
               *[x 
                for x in listaB
                if x not in listaA]]



class CountingClicker:
  """A class can/should have a docstring, just like a function"""

  # these are the member functions
  # every one takes a first parameter "self" (another convention)
  def __init__(self, count = 0):
    """This is the constructor."""
    self.count = count

  def __repr__(self):
    return f"CountingClicker(count={self.count})"

  def click(self, num_times = 1):
      """Click the clicker some number of times."""
      self.count += num_times

  def read(self):
      return self.count

  def reset(self):
      self.count = 0
      
    








