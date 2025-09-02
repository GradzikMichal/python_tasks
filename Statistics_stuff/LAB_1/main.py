import nltk
nltk.download('treebank')

from nltk.corpus import stopwords
from stemming.porter2 import stem
from nltk.book import text7, text9
from string import punctuation


def cesar_cipher(string, d):
    for letter in string:
        if letter != " ":
            asci = ord(letter)
            if asci >= 65 and asci <= 90:
                string = string.translate(str.maketrans(letter, chr(((ord(letter) + d - 65) % 26) + 65)))
            else:
                string = string.translate(str.maketrans(letter, chr(((ord(letter) + d - 97) % 26) + 97)))
    return string


txt = "Ala ma kota"
d = 27
print(cesar_cipher(txt, d))


def sort_dict(d):
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True))


def bookReformat(filename):
    with open(filename, "r") as f:
        words = [word.translate(str.maketrans('', '', punctuation)) for line in f for word in line.split()]
        stopword = set(stopwords.words('english'))
        words = [w for w in words if w not in stopword]
        words = [stem(word) for word in words]
        word_count = {}
        for word in set(words):
            word_count[word] = words.count(word)
        word_count = sort_dict(word_count)
        words = {}
        for word in word_count:
            if word_count[word] >= 100:
                words[word] = word_count[word]
        words = dict(sorted(words.items(), key=lambda item: item[0]))
        return words


def Sunday():
    walt_street = text7.count("Sunday")
    the_man = text9.count("Sunday")
    print(walt_street, the_man)


Sunday()
