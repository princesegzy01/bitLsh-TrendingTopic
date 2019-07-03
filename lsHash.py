from nltk.stem import LancasterStemmer, WordNetLemmatizer
import inflect
import unicodedata
import string
import pylab
import plotly.graph_objs as go
import plotly.plotly as py
import numpy as np
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.util import ngrams
import sys
from nltk.corpus import stopwords
from ELocalitySensitiveHashing import *
import itertools
from collections import Counter
from statistics import mode
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from SetSimilaritySearch import all_pairs
# from LocalitySensitiveHashing import *
import csv
import pandas as pd
from nltk.tokenize import word_tokenize
import re
from nltk import pos_tag, chunk
import nltk
import time

accepted_pos = ['NN', 'NNP', 'NNS', 'NNPS']


# custom functions
def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode(
            'ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    # words = replace_numbers(words)
    words = lemmatize_verbs(words)
    words = remove_stopwords(words)
    return words


bag_of_words = TfidfVectorizer(ngram_range=(2, 2))
# bag_of_words = CountVectorizer(ngram_range=(2, 2))

# df = pd.read_csv('movie_reviews.csv')
df = pd.read_csv('ds.csv')

tweets_df = df.review[:2000]


print("size of dataset : ", tweets_df.shape)
processedTweets = []
tokenizedTweets = []


# print(" >>>>>>>>>>>>>>>>>> : Preprocessing Tweet")
start = time.time()

for index, tweet in enumerate(tweets_df):

    tweet = re.sub(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', tweet)

    tweet = re.sub("&amp;", "", tweet)
    tweet = re.sub("br", "", tweet)
    words = word_tokenize(tweet)
    words = normalize(words)

    newDoc = ' '.join(words)

    processedTweets.append(newDoc)

    tokenizedTweets.append(words)

dfNew = pd.DataFrame(columns=['review'], data=processedTweets)

XA = bag_of_words.fit_transform(processedTweets).toarray()

end = time.time()
diff = end - start
# print(diff, " : seconds ")
# print(" >>>>>>>>>>>>>>>>>> : Min Hashing")


permutation_list = [64, 128, 256, 512]
for num_perms in permutation_list:

    start = time.time()

    minHashArray = []

    for document in XA:
        mhash = MinHash(num_perm=num_perms, seed=3)
        mhash.update(document)
        minHashArray.append(mhash.hashvalues)

    exportedList = []
    for index, arr in enumerate(minHashArray):
        a = arr.tolist()
        a.insert(0, "doc_"+str(index))
        # a = np.insert(arr,  0, str(index) + "_doc", axis=0)
        exportedList.append(a)

    datafile = "csvfile.csv"

    # Export list to csv file
    with open(datafile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(exportedList)

    end = time.time()
    diff = end - start
    # print(diff, " : seconds ")

    print(" >>>>>>>>>>>>>>>>>> : LSH")
    start = time.time()

    print("")
    print("================================")
    print("")
    print(" >>>>>>> Number of permutations : ",  num_perms)

    lsh = LocalitySensitiveHashing(
        datafile=datafile,
        dim=num_perms,
        r=50,
        b=100,
        expected_num_of_clusters=5,
    )
    lsh.get_data_from_csv()
    lsh.initialize_hash_store()
    lsh.hash_all_data()
    similarity_groups = lsh.lsh_basic_for_neighborhood_clusters()
    coalesced_similarity_groups = lsh.merge_similarity_groups_with_coalescence(
        similarity_groups)
    merged_similarity_groups = lsh.merge_similarity_groups_with_l2norm_sample_based(
        coalesced_similarity_groups)

    end = time.time()
    diff = end - start
    print(diff, " : seconds ")

    # print("lsh bucket : ", len(merged_similarity_groups))
    total_doc_number = []
    all_pairs_container = []

    max_buckets = max(merged_similarity_groups, key=len)
    # print("total max bucket length : ", len(max_buckets))
    # print("--")

    all_max_bucket_pairs = []
    for bucket_doc in max_buckets:
        doc_num = bucket_doc.split("_")[1]
        all_max_bucket_pairs.append(processedTweets[int(doc_num)].split(" "))

    flat_lsh = itertools.chain.from_iterable(all_max_bucket_pairs)
    flatLshList = list(flat_lsh)

    filtered_lsh_token = []

    for tokens in flatLshList:
        pos = nltk.pos_tag(word_tokenize(tokens))

        # get token name
        if pos == []:
            continue

        token_name = pos[0][0]

        # get pos name
        pos_name = pos[0][1]

        if pos_name in accepted_pos:
            filtered_lsh_token.append(token_name)

    # print(filtered_lsh_token)
    mostLshWord = dict(zip(*np.unique(filtered_lsh_token, return_counts=True)))
    print("LSH Clustering Most word : ", Counter(mostLshWord).most_common(2))

    for i, bucket in enumerate(merged_similarity_groups):

        if (len(bucket) == 1):
            print("exiting bucket #", i, " because len is 1")
            continue

        doc_num_list = []

        bucket_pairs = []
        for document in bucket:
            doc_num = document.split("_")[1]

            doc_num_list.append(doc_num)

            bucket_pairs.append(processedTweets[int(doc_num)].split(" "))

        all_pairs_container.append(bucket_pairs)
        total_doc_number.append(doc_num_list)

    # print("++++++++++++++++++++++++++++")
    # print(all_pairs_container)
    # sys.exit(0)
    # print("++++++++++++++++++++++++++++")

    total_doc = set()
    total_list_doc = []

    # print(" >>>>>>>>>>>>>>>>>> : Cosine Similarity")
    start = time.time()

    for index, bucket_pairs in enumerate(all_pairs_container):

        child_set = set()

        # The input sets must be a Python list of iterables (i.e., lists or sets).
        # [[1, 2, 3], [3, 4, 5], [2, 3, 4], [5, 6, 7]]
        sets = all_pairs_container[index]
        # all_pairs returns an iterable of tuples.

        # print("**********************************")
        # print(sets)
        # print("**********************************")

        pairs = all_pairs(sets, similarity_func_name="cosine",
                          similarity_threshold=0.75)
        total_tuples = list(pairs)

        for tup_data in total_tuples:
            total_doc.add(total_doc_number[index][tup_data[0]])

    # print("Total doc length : ", len(total_doc))

    feature_list = []
    for d in total_doc:
        feature_list.append(XA[int(d)])

    end = time.time()
    diff = end - start
    # print(diff, " : seconds ")

    pca = PCA(n_components=2).fit(feature_list)
    pca_2d = pca.transform(feature_list)

    X = pca_2d

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # print(" >>>>>>>>>>>>>>>>>> : DBSCAN Clustering")
    start = time.time()

    # cluster the data into five clusters
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    clusters = dbscan.fit_predict(X_scaled)

    # print(" >>>>> cluster size:  ", set(clusters))

    unique_cluster = set(clusters)

    if len(unique_cluster) == 1:
        print("No cluster available, just noise")
        sys.exit(0)

    color_code = ['r', 'g', 'b', 'y', 'w', 'o', 'v', 'p']
    # marker_code = ['+', '*', 'o', 'h', 'p', '1', '2', '3']
    marker_code = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']

    cluster_dict = {}
    for l in unique_cluster:
        cluster_dict[l] = {"document": [], "x": [], "y": []}

    for i in range(pca_2d.shape[0]):
        cluster_dict[dbscan.labels_[i]]["document"].append(i)
        cluster_dict[dbscan.labels_[i]]["x"].append(pca_2d[i, 0])
        cluster_dict[dbscan.labels_[i]]["y"].append(pca_2d[i, 1])

    # print("============================================================")
    # print(" >>>>>> Cluster dict : ", cluster_dict)
    # for i in range(pca_2d.shape[0]):

    #     if dbscan.labels_[i] == 0:
    #         c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
    #     elif dbscan.labels_[i] == 1:
    #         c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
    #     elif dbscan.labels_[i] == -1:
    #         c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')

    # print(dbscan.labels_)
    # print("=====================")

    # filter(lambda a: a != -1, dbscan.labels_)
    removeNoise = [y for y in dbscan.labels_ if y != -1]
    c = Counter(removeNoise)
    mostCluster = c.most_common(1)

    # print("Cluster with most document : ", mostCluster)

    documentKey = mostCluster[0][0]
    documentCount = mostCluster[0][1]

    # print("Cluster dictionary : ")

    most_cluster_documents = cluster_dict.get(documentKey).get('document')

    # print("Most cluster Document : ", most_cluster_documents)

    most_cluster_documents_docs = []

    for doc_id in most_cluster_documents:
        most_cluster_documents_docs.append(tokenizedTweets[doc_id])

    end = time.time()
    diff = end - start
    # print(diff, " : seconds ")

    print(" >>>>>>>>>>>>>>>>>> : Trending Topic")
    start = time.time()

    # print("doc key : ", documentKey, " -- ", "doc count : ", documentCount)

    flat = itertools.chain.from_iterable(most_cluster_documents_docs)
    flatList = list(flat)

    filtered_token = []
    for tokens in flatList:
        pos = nltk.pos_tag([tokens])

        # get token name
        token_name = pos[0][0]

        # get pos name
        pos_name = pos[0][1]

        if pos_name in accepted_pos:
            # print(token_name, " -- ", pos_name)
            filtered_token.append(token_name)

    # print(filtered_token)
    # sys.exit(0)
    mostWord = dict(zip(*np.unique(filtered_token, return_counts=True)))
    print("DBSCAN Clustering Most word : ", Counter(mostWord).most_common(2))

    end = time.time()
    diff = end - start
    # print(diff, " : seconds ")
    # sys.exit(0)

    # for index, key in enumerate(cluster_dict.keys()):
    # print(cluster_dict[key])

    # x = np.asarray(list(cluster_dict[key]["x"]))
    # y = np.asarray(list(cluster_dict[key]["y"]))

    # print(x.shape)
    # print(y.shape)
    # plt.scatter(x, y, c=color_code[index], marker=marker_code[index])

    # sys.exit(0)

    # plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2', 'Noise'])
    # plt.legend([k for k in legend], [k for k in legend])
    # print(legend)
    # plt.title('DBSCAN finds 2 clusters and noise')
    # plt.show()

    # print("")
    # print(colors)
