from __future__ import division
import itertools
import re
import hashlib

# a shingle in this code is a string with K-words
K = 4

def jaccard_set(s1, s2):
    " takes two sets and returns Jaccard coefficient"
    u = s1.union(s2)
    i = s1.intersection(s2)
    return len(i)/len(u)

def make_a_set_of_tokens(doc):
    """makes a set of K-tokens"""

    # replace non-alphanumeric char with a space, and then split
    tokens = re.sub("[^\w]", " ",  doc).split()

    sh = set()
    for i in range(len(tokens)-K):
        t = tokens[i]
        for x in tokens[i+1:i+K]:
            t += ' ' + x 
        sh.add(t)
    return sh

if __name__ == '__main__':

    documents = [
      "The legal system is made up of civil courts, criminal courts and specialty courts such as family law courts and bankruptcy court. Each court has its own jurisdiction, which refers to the cases that the court is allowed to hear. In some instances, a case can only be heard in one type of court. For example, a bankruptcy case must be heard in a bankruptcy court. In other instances, there may be several potential courts with jurisdiction. For example, a federal criminal court and a state criminal court would each have jurisdiction over a crime that is a federal drug offense but that is also an offense on the state level.",
      "The legal system is comprised of criminal and civil courts and specialty courts like bankruptcy and family law courts. Every one of the courts is vested with its own jurisdiction. Jurisdiction means the types of cases each court is permitted to rule on. Sometimes, only one type of court can hear a particular case. For instance, bankruptcy cases an be ruled on only in bankruptcy court. In other situations, it is possible for more than one court to have jurisdiction. For instance, both a state and federal criminal court could have authority over a criminal case that is illegal under federal and state drug laws.",
      "In many jurisdictions the judicial branch has the power to change laws through the process of judicial review. Courts with judicial review power may annul the laws and rules of the state when it finds them incompatible with a higher norm, such as primary legislation, the provisions of the constitution or international law. Judges constitute a critical force for interpretation and implementation of a constitution, thus de facto in common law countries creating the body of constitutional law."]
    
    # print(documents)
    shingles = []
    # handle documents one by one
    # makes a list of sets which are compresized of a list of K words string
    for doc in documents:
        # makes a set of tokens
        # sh = set([' ', ..., ' '])
        sh = make_a_set_of_tokens(doc)

        # print("sh = %s", sh)

        # hasing
        bucket = map(hash, sh)

        # print("bucket = %s", set(bucket))
        
        # shingles : list of sets (sh)
        shingles.append(set(bucket))

    # print("shiingles : ", len(shingles[0]))
    # print("doc : ", sh)


    combinations = list( itertools.combinations([x for x in range(len(shingles))], 2) )
    print("combinations : ", combinations)

    # # compare each pair in combinations tuple of shingles
    # for c in combinations:
    #     i1 = c[0]
    #     i2 = c[1]
    #     jac = jaccard_set(shingles[i1], shingles[i2])
    #     print("%s : jaccard=%s", c,jac)
