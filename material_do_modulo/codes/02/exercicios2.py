# Utilize List Comprehension para elevar ao quadrado 
# todos os elementos de uma lista;
nums = [1,2,3,4]

[x*x for x in nums]

# Utilize List Comprehension para retirar elementos 
# palíndromos de uma lista de strings;

def eh_palindromo(s):
    return(s == s[::-1])

strings = ["ana", "ia", "teste", "subinoonibus", "aaaaaa"]

[p for p in strings if not eh_palindromo(p)]


# Utilize List Comprehension para, dadas as listas A e B, 
# criar uma lista C composta apenas pelos elementos presentes 
# em ambas A e B;
a = [1,2,3,4,5]
b = [3,4,5,6,7]

[x for x in set(a+b) if x in a and x in b]

