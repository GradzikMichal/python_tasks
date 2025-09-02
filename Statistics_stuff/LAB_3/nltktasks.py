import nltk
from nltk.corpus import gutenberg
from nltk.book import text1, text2, text3, text4, text5, text6, text7, text8, text9
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from stemming.porter2 import stem
# Load the text corpus
# nltk.download('gutenberg')
from string import punctuation
import networkx as nx
import matplotlib.pyplot as plt
import multiset


def lev(a, b):
    if len(b) == 0:
        return len(a)
    elif len(a) == 0:
        return len(b)
    elif a[0] == b[0]:
        return lev(a[1:], b[1:])
    else:
        return 1 + min(lev(a[1:], b), lev(a, b[1:]), lev(a[1:], b[1:]))


def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        return 0
    return sum(s1[i] != s2[i] for i in range(len(s1)))


def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0


def preprocessing(text):
    words = list(word.translate(str.maketrans('', '', punctuation)) for line in text for word in line.split())
    stopword = set(stopwords.words('english'))
    words = [w.lower() for w in words if w not in stopword]
    words = [stem(word) for word in words]
    return words


def diameterMS():
    """
       Returns a diameter of the metric space which is a set of words of text1.

       Args:
       -

       Returns:
       - A diameter of the metric space which is a set of words of text1.
    """
    words = {word.translate(str.maketrans('', '', punctuation)) for line in text1 for word in line.split()}
    stopword = set(stopwords.words('english'))
    words = [w.lower() for w in words if w not in stopword]
    words = list(set([stem(word) for word in words]))

    # Compute the diameter of the metric space (S, d)
    diameter = 0
    for x in words:
        for y in words:
            distance = hamming_distance(x, y)
            if distance > diameter:
                diameter = distance
    print("Diameter of metric space: " + str(diameter))
    return diameter


def Jaccard_consecutive():
    """
           Construct a dictionary that assigns each pair of consecutive words in text1 the Jaccard similarity between them

           Args:
           -

           Returns:
           - A dictionary in which key is a pair of consecutive words and value is a Jaccard similarity
    """
    dictionary = {}
    words = list(word.translate(str.maketrans('', '', punctuation)) for line in text1 for word in line.split())
    stopword = set(stopwords.words('english'))
    words = [w.lower() for w in words if w not in stopword]
    words = [stem(word) for word in words]
    for i in range(len(words) - 1):
        word1 = multiset.Multiset(words[i])
        word2 = multiset.Multiset(words[i + 1])
        similarity = jaccard_similarity(word1, word2)
        dictionary[(words[i], words[i + 1])] = similarity
    print("Dictionary: ")
    for key, value in dictionary.items():
        print(key, value)
    return dictionary


def Jaccard_of_texts():
    """
        Construct a dictionary that assigns each pair of text1-text9 the Jaccard similarity between them

        Args:
           -

        Returns:
           - A dictionary in which key is a pair of texts and value is a Jaccard similarity between them
    """
    dictionary = {}
    texts = [
        multiset.Multiset(preprocessing(text1)),
        multiset.Multiset(preprocessing(text2)),
        multiset.Multiset(preprocessing(text3)),
        multiset.Multiset(preprocessing(text4)),
        multiset.Multiset(preprocessing(text5)),
        multiset.Multiset(preprocessing(text6)),
        multiset.Multiset(preprocessing(text7)),
        multiset.Multiset(preprocessing(text8)),
        multiset.Multiset(preprocessing(text9))]
    for i in range(len(texts) - 1):
        for j in range(i + 1, len(texts)):
            similarity = jaccard_similarity(texts[i], texts[j])
            dictionary[("text" + str(i + 1), "text" + str(j + 1))] = similarity
    print("Dictionary: ")
    for key, value in dictionary.items():
        print(key, value)
    return dictionary


def get_words_within_distance():
    """
    Returns a list of words in the given text that have a Levenshtein distance
    from the given word less than or equal to the given distance.

    Args:
        -

    Returns:
    - A list of words in the given text with distance less than or equal to the given distance.
    """
    words = [word.translate(str.maketrans('', '', punctuation)) for line in text1 for word in line.split()]
    stopword = set(stopwords.words('english'))
    words = [w.lower() for w in words if w not in stopword]
    words = [stem(word) for word in words]
    text, word, distance = words, "dog", 3
    words_within_distance = []
    for w in set(text):
        if len(w) < len(word) + distance:
            if lev(w, word) <= distance:
                words_within_distance.append(w)
    print("Words within distance: \n", words_within_distance)
    return words_within_distance


def releditdist():
    """
        COMPUTATION IS LONG!
        Find two different words in text2 with the minimal relative edit distance.

    Args:
        -

    Returns:
    - Two different words and minimal relative edit distance between them.
    """
    words = [word.translate(str.maketrans('', '', punctuation)) for line in text2 for word in line.split()]
    stopword = set(stopwords.words('english'))
    words = [w.lower() for w in words if w not in stopword]
    words = list(set(stem(word) for word in words))
    word1 = ''
    word2 = ''
    minimal = 100.0
    for i in range(0, len(words) - 2):
        for j in range(i + 1, len(words) - 1):
            v = words[i]
            w = words[j]
            print(v, w)
            if v != w:
                rel_e_dist = lev(v, w) / (len(v) + len(w))
                if rel_e_dist < minimal:
                    minimal = rel_e_dist
                    word1 = v
                    word2 = w
    print("Words with minimal relative edit distance: "+str(word1)+" "+str(word1))
    return word1, word2, minimal


def graph(l, s):
    """
        Draw a graph with nodes labeled by words in text2 that appear at least l times. Add edges connecting
        pairs of words with the Levenshtein distance smaller than s.

        Args:
        - l: number of times word must appear in text
        - s: Levenshtein distance

        Returns:
        - Drawing graph of words
    """
    words = list(word.translate(str.maketrans('', '', punctuation)) for line in text2 for word in line.split())
    stopword = set(stopwords.words('english'))
    words = [w.lower() for w in words if w not in stopword]
    words = [stem(word) for word in words]
    c_words = []
    for w in set(words):
        number = words.count(w)
        if number >= l:
            c_words.append(w)
    dist = {}
    print(len(c_words), c_words)
    for i in range(0, len(c_words) - 2):
        for j in range(i + 1, len(c_words) - 1):
            d = lev(c_words[i], c_words[j])
            if d < s:
                dist[(c_words[i], c_words[j])] = d
    G = nx.Graph()

    for (key, element) in dist.items():
        G.add_nodes_from(key)
        G.add_edge(*key)
    pos = nx.circular_layout(G)
    nx.draw(
        G, pos, edge_color='black', width=1, linewidths=1,
        node_size=500, node_color='pink', alpha=0.9,
        labels={node: node for node in G.nodes()}
    )
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=dist,
        font_color='red'
    )
    plt.axis('off')
    plt.show()


#diameterMS()
#Jaccard_consecutive()
#Jaccard_of_texts()
#get_words_within_distance()
#releditdist()
#graph(230, 4)
