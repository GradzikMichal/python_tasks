from functools import reduce
import multiset


def sim_bitstrings(b, n):
    """
    Returns a list of all nonempty bitstrings b', such that the Hamming distance between b and b' is equal to n and n is bigger than 0.
    b' is equal length of b

    Args:
        b (str): The input bitstring.
        n (int): The desired Hamming distance between b and b'. Bigger than 0.

    Returns:
        A list of all nonempty bitstrings b', such that the Hamming distance between b and b' is equal to n.
        Assumes that both bitstrings have the same length.
    """
    m = len(b)
    result = []
    for i in range(2 ** m):
        bi = bin(i)[2:].zfill(m)
        dist = sum(1 for x, y in zip(b, bi) if x != y)
        if dist == n:
            result.append(bi)
    return result


def set_jaccard(x, y):
    """
    Computes the Jaccard similarity for two nonempty sets x and y.

    Args:
        x (set): The first nonempty set.
        y (set): The second nonempty set.

    Returns:
        The Jaccard similarity between x and y, defined as the size of the intersection
        of x and y divided by the size of the union of x and y.
        Returns None if at least one of the sets is empty.
    """
    if not x or not y:
        return None
    intersection = len(x.intersection(y))
    union = len(x.union(y))
    jaccard = intersection / union
    return jaccard


def bag_jaccard(x, y):
    """
    Computes the Jaccard similarity for two nonempty strings x and y treated as bags of words.

    Args:
        x (str): The first nonempty string.
        y (str): The second nonempty string.

    Returns:
        The Jaccard similarity between x and y, defined as the size of the intersection
        of the multisets of words in x and y divided by the size of the union of the multisets of words
        in x and y.
    """
    x_words = multiset.Multiset(x.split())
    y_words = multiset.Multiset(y.split())
    intersection = len(x_words & y_words)
    union = len(x_words | y_words)
    jaccard = intersection / union
    return jaccard


def shingles(s, k):
    """
    Returns a nonempty set of all k-shingles for a given nonempty string s.

    Args:
        s (str): The input nonempty string without punctuation. Otherwise raise error.
        k (int): The length of each shingle. Bigger then 1. Otherwise raise error.

    Returns:
        A set of all k-shingles of s.
    """
    # Check that k is a positive integer
    if not isinstance(k, int) or k < 1:
        raise ValueError('k must be a positive integer')

    # Check that s is a string
    if not isinstance(s, str):
        raise ValueError('s must be a string')

    # Create a set of all k-shingles of s
    shingles_set = set()
    words = s.split()
    for i in range(0, len(words) - k + 1):
        shingle = words[0 + i:k + i]
        shingles_set.add(' '.join(shingle))
    return shingles_set


def diameter(S, d):
    """
    Computes the diameter of a metric space (S, d).

    Args:
        S (set): The nonempty set of points in the metric space.
        d (function): The distance function on S.

    Returns:
        The diameter of (S, d), defined as the maximum distance between any two points in S.
    """
    max_distance = 0
    for x in S:
        for y in S:
            distance = d(x, y)
            if distance > max_distance:
                max_distance = distance
    return max_distance


def hamming_distance(s1, s2):
    """
    Computes the Hamming distance between two nonempty strings s1 and s2.

    Args:
        s1 (str): The first string.
        s2 (str): The second string.

    Returns:
        The Hamming distance between s1 and s2, defined as the number of positions
        at which the corresponding symbols are different. If length of two strings are different raises error.
    """
    if len(s1) != len(s2):
        raise ValueError('s1 and s2 must have the same length')
    return sum(s1[i] != s2[i] for i in range(len(s1)))
