import sys

from datasketch import MinHash, MinHashLSH

# set1 = set(['Number','one','bestselling','author','and', 'former','ex-President','President'])

# set2 = set(['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
#             'estimating', 'the', 'similarity', 'between', 'documents'])
# set3 = set(['minhash', 'is', 'probability', 'data', 'structure', 'for',
#             'estimating', 'the', 'similarity', 'between', 'documents'])


set1 = set(['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
            'estimating', 'the', 'similarity', 'between', 'datasets'])
set2 = set(['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
            'estimating', 'the', 'similarity', 'between', 'documents'])
set3 = set(['minhash', 'is', 'probability', 'data', 'structure', 'for',
            'estimating', 'the', 'similarity', 'between', 'documents'])
set4 = set(['how', 'can', 'win', 'election', 'government', 'country'])
set5 = set(['country', 'bad', 'government', 'win'])


print(set1)
sys.exit(0)
# m1 = MinHash(num_perm=128)
# m2 = MinHash(num_perm=128)
# m3 = MinHash(num_perm=128)
# for d in set1:
#     m1.update(d.encode('utf8'))
# for d in set2:
#     m2.update(d.encode('utf8'))
# for d in set3:
#     m3.update(d.encode('utf8'))

# Create LSH index
# lsh = MinHashLSH(threshold=0.2, num_perm=128, params = (2,3))

k_sig = 1024

m1 = MinHash(num_perm=k_sig)
m2 = MinHash(num_perm=k_sig)
m3 = MinHash(num_perm=k_sig)
m4 = MinHash(num_perm=k_sig)
m5 = MinHash(num_perm=k_sig)

for d in set1:
    m1.update(d.encode('utf8'))
for d in set2:
    m2.update(d.encode('utf8'))
for d in set3:
    m3.update(d.encode('utf8'))
for d in set4:
    m4.update(d.encode('utf8'))
for d in set5:
    m5.update(d.encode('utf8'))

# Create LSH index
lsh = MinHashLSH(threshold=0.9, num_perm=k_sig, params=(2, 1))
lsh.insert("m1", m1)
lsh.insert("m2", m2)
lsh.insert("m3", m3)
lsh.insert("m4", m5)
lsh.insert("m5", m5)

# result = lsh.query(m1)
# print("Approximate neighbours with Jaccard similarity > 0.7", result)

# print(len(lsh.hashtables))
# print(lsh.hashtables)


# print(lsh.get_counts())

# print("")
for bucketNumber, table in enumerate(lsh.hashtables):

    # print("******")
    # print(len(table.keys()))

    # print("******")
    # print(dir(table))
    print("================ Bucket # ", bucketNumber, " ===============")
    # print(table._dict)
    # print(table.keys())
    for key in table.keys():
        # print(key)
        print(table.get(key))
    # print("----")
