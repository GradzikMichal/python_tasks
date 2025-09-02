import random
import string
import hashlib
from copy import deepcopy

import numpy as np
from numba import njit
from numpy.random import seed
from numpy.random import randint

import nltk
from nltk.corpus import gutenberg
from nltk.book import text1, text2, text3
from nltk.tokenize import RegexpTokenizer
from string import punctuation
from nltk.corpus import stopwords
from stemming.porter2 import stem
import multiset


def gen_hash(m):
    '''
    Returns a function which takes bitstring and returns bitstring of length m

    Args:
        m (int): The length of output bitstring. Non-zero and bigger than zero

    Returns:
        A bitstring of length m
    '''
    return lambda x: bin(abs(hash(x)))[2:m + 2]


def preprocessing(text):
    words = list(word.translate(str.maketrans('', '', punctuation)) for line in text for word in line.split())
    stopword = set(stopwords.words('english'))
    words = [w.lower() for w in words if w not in stopword]
    words = {stem(word) for word in words}
    return words


def preprocessing8(text):
    words = list(word.translate(str.maketrans('', '', punctuation)) for line in text for word in line.split())
    stopword = set(stopwords.words('english'))
    words = [w.lower() for w in words if w not in stopword]
    words = [stem(word) for word in words]
    words = {word for word in words if len(word) < 8}
    return words


def task2(n):
    '''
    Returns a minimal hash of concatenated 2 bit strings. The maximal n for me was 1000.

    Args:
        n (int): Non-zero and bigger than zero. Number of bit strings generated

    Returns:
        A minimal value of hash of randomly generated bit strings
    '''
    S = set()
    if n < 2:
        print("n is to small")
        return False
    while len(S) < n:
        S.add(''.join(random.choice(['0', '1']) for i in range(100)))
    S = list(S)
    to_check = S[0] + S[0]
    m = hashlib.sha1()
    m.update(to_check.encode())
    min = m.hexdigest()
    for x in range(0, len(S) - 2):
        for y in range(x + 1, len(S) - 1):
            to_check = S[x] + S[y]
            m = hashlib.sha1()
            m.update(to_check.encode())
            value = m.hexdigest()
            if value < min:
                min = value
    print(min)
    return min


def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0


def task3and4():
    '''
        Printing estimated Jaccard similarity and exact (true) Jaccard similarity.

        Args:
            No input arguments

        Returns:
            Nothing.
            The function is printing estimated and exact Jaccard similarity.

        Sidenotes:
        In task was to do 100 minhashes but making 100 permutations is mathematically the same but saves some space.
        Random mapping is used for reducing the dimensionality of a data. Result will be very similar
    '''
    set1 = preprocessing8(text1)
    set2 = preprocessing8(text2)
    set3 = preprocessing8(text3)
    words = set.union(set1, set2, set3)
    table = []
    # hashes = [lambda x: random.randint(0, 10000) * x + x + 4] * 100
    hashes = []
    for x in words:
        word = [x]
        if x in set1:
            word.append(1)
        else:
            word.append(0)
        if x in set2:
            word.append(1)
        else:
            word.append(0)
        if x in set3:
            word.append(1)
        else:
            word.append(0)
        table.append(word)

    for h in range(100):
        random.shuffle(table)
        hash = []
        for i in range(3):
            for x in range(len(table)):
                if table[x][i + 1]:
                    hash.append(table[x][0])
                    break
        hashes.append(hash)
    sum = 0
    for h in hashes:
        if h[0] == h[1]:
            sum += 1
    J_1_2 = sum / float(100)
    sum = 0
    for h in hashes:
        if h[0] == h[2]:
            sum += 1
    J_1_3 = sum / float(100)
    sum = 0
    for h in hashes:
        if h[1] == h[2]:
            sum += 1
    J_3_2 = sum / float(100)

    print("Estimated Jaccard of set1 and set2 sim: ", J_1_2)
    print("True Jaccard of set1 and set2 sim: ", jaccard_similarity(multiset.Multiset(set1), multiset.Multiset(set2)))
    print("Estimated Jaccard of set1 and set3 sim: ", J_1_3)
    print("True Jaccard of set1 and set3 sim: ", jaccard_similarity(multiset.Multiset(set1), multiset.Multiset(set3)))
    print("Estimated Jaccard of set2 and set3 sim: ", J_3_2)
    print("True Jaccard of set3 and set2 sim: ", jaccard_similarity(multiset.Multiset(set3), multiset.Multiset(set2)))


def task5():
    '''
        Printing combinations of set which Jaccard Similarity is bigger than 0.5

        Args:
            No input arguments

        Returns:
            Nothing.
            The function is printing Jaccard similarity between sets.
    '''
    set1 = list(preprocessing8(text1))
    set1_len = len(set1)
    sets = [set1]
    num_hashes = 220
    for i in range(1, 100):
        words_removed = int((i / 100.0 * set1_len))
        set2 = deepcopy(set1)
        for x in range(words_removed):
            set2.pop(random.randint(0, set1_len - x - 1))
        sets.append(set2)
    hashes = [lambda x: (random.randint(0, 10000) * x + 4) % set1_len] * num_hashes
    hash_val = []
    for i in range(num_hashes):
        hash_row = []
        for j in range(set1_len):
            hash_row.append(hashes[i](j))
        hash_val.append(hash_row)
    table = []
    hash_val = np.column_stack(hash_val)
    for x in set1:
        word = [x]
        for s in sets:
            if x in s:
                word.append(1)
            else:
                word.append(0)
        table.append(word)
    rows = []
    probably_same = []
    columns = []
    t = [[11000 for _ in range(len(sets))] for _ in range(num_hashes)]
    for hash_num in range(len(t)):
        for set_num in range(len(t[0])):
            row = 0
            for row_tab in table:
                if row_tab[set_num+1] == 1:
                    t[hash_num][set_num] = min(t[hash_num][set_num], hash_val[row][hash_num])
                row+=1

    r = 3
    abc = 0
    for row in t:
        rows.append(row)
        if len(rows) == r:
            abc+=1
            columns = np.column_stack(rows)
            for col1 in range(1, len(columns) - 1):
                for col2 in range(col1 + 1, len(columns)):
                    if np.equal(columns[col1], columns[col2]).all():
                        probably_same.append((col1, col2))
            rows = []
            columns = []
    for col1 in range(1, 99):
        for col2 in range(col1 + 1, 100):
            c = probably_same.count((col1, col2))
            if c / float(abc) > 0.5:
                print(str(col1), str(col2), c / float(abc))


task3and4()