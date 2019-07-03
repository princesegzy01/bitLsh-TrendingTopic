from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
print(vectorizer.get_feature_names())
# ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
print(X.shape)
print(X[0])


m = MinHash(num_perm=2, seed=3)

m.update(X[0])

print(m.hashvalues)
