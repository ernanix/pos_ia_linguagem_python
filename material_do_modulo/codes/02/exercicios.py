def segundos(h, m, s):
    """retorna um valor de horas, minutos e segundos em
    segundos"""
    return(h*3600+m*60+s)

assert segundos(1,0,0) == 3600
print(segundos(2,0,1))

def eh_palindromo(s):
    """retorna true se a string s é um palindromo"""
    i = 0
    size = len(s)//2
    while i < size:
        if s[i] != s[-i-1]:
            return(False)
        i += 1
    return(True)

def eh_palindromo2(s):
    return(s == s[::-1])

def soma_de_quadrados(nums):
    """retorna a soma dos quadrados dos elementos de uma lista"""
    soma = 0
    for num in nums:
        soma += num*num

    return(soma)


def fatorial(n):
    """calcula recursivamente o fatorial de um número inteiro"""
    if n == 0:
        return(1)
    else:
        return(n*fatorial(n-1))


def inverte_lista(lista):
    """inverte recursivamente uma lista l"""
    if lista:
        return([lista[-1]] + inverte_lista(lista[0:-1]))
    else:
        return([])
