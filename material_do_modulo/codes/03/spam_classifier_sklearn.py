from collections import Counter, defaultdict
from machine_learning import split_data
import math, random, re, glob
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

def tokenize(message):
    message = message.lower()                       # convert to lowercase
    all_words = re.findall("[a-z0-9']+", message)   # extract the words
    return set(all_words)                           # remove duplicates


def count_words(training_set):
    """training set consists of pairs (message, is_spam)"""
    counts = defaultdict(lambda: [0, 0])
    for message, is_spam in training_set:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
    return counts

def get_subject_data(path):

    data = []

    # regex for stripping out the leading "Subject:" and any spaces after it
    subject_regex = re.compile(r"^Subject:\s+")

    # glob.glob returns every filename that matches the wildcarded path
    for fn in glob.glob(path):
        is_spam = "ham" not in fn

        with open(fn,'r',encoding='ISO-8859-1') as file:
            for line in file:
                if line.startswith("Subject:"):
                    subject = subject_regex.sub("", line).strip()
                    data.append((subject, is_spam))

    return data

class Classifier:
    
    used_features= []
    gnb = None
    
    def __init__(self):
        self.used_features = []
        self.gnb = GaussianNB()
    
    def msgs_to_data_frame(self, data):
        r = list()
        for message, is_spam in data:
            words = list(tokenize(message))
            d = {word: 1 for word in words}
            d.update({'__is_spam': is_spam, '__message': message})
            r.append(d)
            
        return(pd.DataFrame(r).fillna(0))

    
    def train(self, data):
        self.used_features = data.columns.drop(['__is_spam','__message'])
        self.gnb.fit(data[self.used_features], data['__is_spam'])
    
    def predict(self, msg):
        words = tokenize(msg)
        words = [w for w in words if w in self.used_features]
        arr = pd.Series(index=self.used_features).fillna(0)
        arr[words] = 1
        return(self.gnb.predict([arr])[0])
        
msgs = get_subject_data(r"./emails/*/*")

c = Classifier()

data = c.msgs_to_data_frame(msgs)

X_train, X_test = train_test_split(data, test_size=0.25, random_state=1)

c.train(X_train)

classified = [(msg, is_spam, c.predict(msg))
              for msg, is_spam in X_test[['__message','__is_spam']].values]

counts = Counter((is_spam, predict) # (actual, predicted)
                     for _, is_spam, predict in classified)