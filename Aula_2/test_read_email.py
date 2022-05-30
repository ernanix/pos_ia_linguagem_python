from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from typing import TypeVar, List, Tuple, Dict, Iterable, NamedTuple, Set
import re
import nltk

ps = PorterStemmer()

with open("texto.txt", "r", encoding=("UTF-8")) as texto:
    for line in texto:
        if line.startswith("Subject:"):
            break;
    mensagem = line
    for line in texto:
        mensagem = mensagem + line
#print(mensagem)        

mensagem_stm = []            
for w in mensagem.split():
    mensagem_stm.append(ps.stem(w))

print (' '.join(np.unique(mensagem_stm)))    


nltk.download('punkt')
def tokenize(text: str) -> Set[str]:
    text = text.lower()                         # Convert to lowercase,
    all_words = nltk.word_tokenize(text) #re.findall("[a-z0-9']+", text)  # extract the words, and
    return all_words #set(all_words) 

jhon = tokenize("Teste Testa Teste'3 Tiesto5 Tiete Testa")

"""
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()
         
         #Now, let's choose some words with a similar stem, like:
         
example_words =["python","pythoner","pythoning","pythoned","pythonly"]
         
         #Next, we can easily stem by doing something like:
         
for w in example_words:
    print(ps.stem(w))
"""