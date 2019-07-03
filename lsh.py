import pandas as pd
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
import sys
import re
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from datasketch import MinHash, MinHashLSH
import numpy as np

# bag_of_words = CountVectorizer(ngram_range=(2, 2))
bag_of_words = TfidfVectorizer(ngram_range=(2, 2))

df = pd.read_csv('movie_reviews.csv')
tweets = df.review[:50]

stopwordList = stopwords.words("english")

processedTweets = []
for index, tweet in enumerate(tweets):

    # tweet = re.sub("< br / >", " ", tweet)

    tweet = tweet.replace('< br / >', ' ')
    document = word_tokenize(tweet)

    tw = [word for word in document if word not in set(stopwordList)]

    newDoc = ' '.join(tw)
    processedTweets.append(newDoc)


dfNew = pd.DataFrame(columns=['review'], data=processedTweets)
# print(dfNew.head(10))
# print(processedTweets[:20])

# X = bag_of_words.fit_transform(processedTweets)
# XD = bag_of_words.fit_transform(processedTweets).todense()

XA = bag_of_words.fit_transform(processedTweets).toarray()
# XA = processedTweets

feature_names = bag_of_words.get_feature_names()


# print(feature_names)
# print(bag_of_words.vocabulary_)

# Create LSH index
lsh = MinHashLSH(threshold=0.5, num_perm=2048, params=(5, 10))

for index, document in enumerate(XA):
    # for index, document in enumerate(processedTweets):

    m = MinHash(num_perm=2, seed=3)

    for token in document:

        # print(token)
        m.update(token)
        # m.update(token.encode('utf-8'))

        print(document)

        sys.exit(0)

    lsh.insert(index, m)
# minhashx.append(m)

# print(minhashx[0])


for bucketNumber, table in enumerate(lsh.hashtables):
    print("================ Bucket # ", bucketNumber, " ===============")
    for key in table.keys():
        print(table.get(key))


print("done")
