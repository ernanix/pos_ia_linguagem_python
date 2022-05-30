
f = open("texto.txt", "r")

print(f.read())


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