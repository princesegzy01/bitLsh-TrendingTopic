from datasketch import MinHash


k_sig = 1024
minhash = MinHash(num_perm=k_sig)
minhash2 = MinHash(num_perm=k_sig)
minhash3 = MinHash(num_perm=k_sig)

minhash.update(
    "This is a good algorithm that can perform wild ranges of services".encode('utf-8'))

minhash2.update(
    "The computing algorithm for this is very cool and works really well".encode('utf-8'))

minhash3.update(
    "The computing algorithm for this is very cool and works really wellx".encode('utf-8'))

# print(minhash.hashvalues)
# print(minhash2.hashvalues)
# print(minhash3.hashvalues)


print(minhash2.jaccard(minhash3))
# print(minhash2.permutations)
# print(minhash3.permutations)
