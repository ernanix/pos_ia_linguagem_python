categories = ['alt.atheism', 'soc.religion.christian',
'comp.graphics', 'sci.med']
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', categories=categories,
                                  shuffle=True, random_state=42)

twenty_train.target_names
twenty_train.target[:10]
for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])
    
    
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape
count_vect.vocabulary_.get(u'algorithm')