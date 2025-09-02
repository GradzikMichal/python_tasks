import random

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from pydtmc import MarkovChain, plot_graph
import networkx as nx
import pybind11
import walker
import csrgraph as cg
from scipy.sparse import diags
from sklearn.preprocessing import normalize
from fast_pagerank import pagerank
from scipy import sparse
import math
import operator
from graphs import plotGraph


def getTrustRank(teleport_set, node_num):
    edges = [(1, 2), (2, 1), (2, 2), (2, 6), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1), (6, 6), (6, 7), (5, 8), (7, 9),
             (8, 9), (5, 1), (4, 1), (6, 11), (11, 7), (7, 10), (10, 1), (8, 12)]
    G = nx.DiGraph()
    G.add_edges_from(edges)
    T = nx.google_matrix(G, alpha=0.9, dangling={12:0, 9:0, 8:0}, nodelist=[1,2,3,4,5,6,7,8,9,10,11,12])
    T[np.isnan(T)] = 0.
    # Defining the Transition matrix
    #v = len(teleport_set) * [1] + (node_num - len(teleport_set)) * [0]
    v = np.zeros(node_num)
    for ts in teleport_set:
        v[ts-1] = 1
    #random.shuffle(v)  # random shuffling the good and the bad
    v  = v / np.sum(v)
    b = 0.9  # value of damping factor
    temp = v
    t = v
    t = b * np.dot(T, t) + (1 - b) * v
    i = 1  # number of iterations

    while np.linalg.norm(temp - t) > 0:  # iterate till the vector converges
        temp = t
        t = b * np.dot(T, t) + (1 - b) * v
        i = i + 1

    trust_rank_node_score = []  # to store the node and the corresponding trustrank score
    for k in range(1, len(t)):
        if t[k] != 0:
            trust_rank_node_score.append([k, t[k]])

    # sort the trustrank scores in the decreasing order
    trust_rank_node_score = sorted(trust_rank_node_score, key=operator.itemgetter(1), reverse=True)
    print("Trust rank: ", trust_rank_node_score)
    PR = nx.pagerank(G, alpha=0.9, max_iter=100000, tol=1e-06, weight=None)
    print("PageRank", PR)
    SpamMass = dict()
    for idx, val in trust_rank_node_score:
        SpamMass[idx] = (PR[idx] - val)/val
    print("SpamMass", SpamMass)
    return trust_rank_node_score


def zad1():
    p = [
        [0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        [1/3, 1/3, 1/3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.2, 0.0, 0.2, 0.2, 0.0, 0.0, 0.2, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    ]
    G = (['1', '2', '3', '4', '5', '7', '8', '9', '10'])
    mc = MarkovChain(p, G)
    print(mc)
    sequence = ["1"]
    for i in range(1, 30):
        current_state = sequence[-1]
        next_state = mc.next(current_state)
        print((' ' if i < 10 else '') + f'{i}) {current_state} -> {next_state}')
        sequence.append(next_state)

def zad2():
        p = [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.0, 0.0, 0.25],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.5]
        ]
        G = ['1', '2', '3', '4', '5', '6']
        mc = MarkovChain(p, G)
        print(mc)
        sequence = ["1"]
        for i in range(1, 22):
            current_state = sequence[-1]
            next_state = mc.next(current_state)
            print((' ' if i < 10 else '') + f'{i}) {current_state} -> {next_state}')
            sequence.append(next_state)

def zad3():
    #https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html
    edges = [(1, 2), (2, 1), (2, 2), (2, 6), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1), (6, 6), (6, 7), (5, 8), (7, 9),
           (8, 9)]
    edgesl = np.array([[0, 1], [1, 0], [1, 1], [1, 5], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [5, 5], [5, 6], [4, 7], [6, 8],
             [7, 8]])
    weights = np.ones(14)
    g = sparse.csr_matrix((weights, (edgesl[:, 0], edgesl[:, 1])), shape=(9, 9))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    PR0 = nx.pagerank(G, alpha=0, max_iter=100, tol=1e-06, weight=None)
    print("Taxation parameter 0")
    print(PR0)
    pr = pagerank(g, p=0.0)
    print(pr)
    print("-------------------------------------------------------------")
    PR01 = nx.pagerank(G, alpha=0.1, max_iter=100, tol=1e-06, weight=None)
    print("Taxation parameter 0.1")
    print(PR01)
    pr = pagerank(g, p=0.1)
    print(pr)
    print("-------------------------------------------------------------")
    PR09 = nx.pagerank(G, alpha=0.9, max_iter=100, tol=1e-06, weight=None)
    print("Taxation parameter 0.9")
    print(PR09)
    pr = pagerank(g, p=0.9)
    print(pr)
    print("-------------------------------------------------------------")
    PR1 = nx.pagerank(G, alpha=1, max_iter=100, tol=1e-06, weight=None)
    print("Taxation parameter 1")
    print(PR1)
    pr = pagerank(g, p=1.0)
    print(pr)
    nx.draw(G, with_labels=True)
    plt.show()


def generating_graphs(n,l):
    edges = []
    for n_i in range(n):
        population = list(range(n))
        del population[n_i]
        lst = random.sample(population=population,k=l)
        for neighbour in lst:
            edges.append((n_i, neighbour))
    return edges

def zad4():
    n = [2, 10, 20, 1000]
    l = [1, 2, 5, 10, 20, 50]
    beta = [0, 0.01, 0.05, 0.1, 0.2]
    for n_i in n:
        for l_i in l:
            if n_i > l_i:
                for beta_i in beta:
                    print("n= "+str(n_i)+" l= "+str(l_i)+" beta= "+str(beta_i))
                    edges = generating_graphs(n_i, l_i)
                    print(edges)
                    G = nx.DiGraph()
                    G.add_edges_from(edges)
                    nx.draw(G, with_labels=True)
                    walks = walker.random_walks(G, n_walks=1, walk_len=10)
                    print(walks)
                    plt.show()


def list4zad1():
    #https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html
    edges = [(1, 2), (2, 1), (2, 2), (2, 6), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1), (6, 6), (6, 7), (5, 8), (7, 9),
           (8, 9), (5, 1), (4, 1), (6, 11), (11, 7), (7, 10), (10, 1), (8, 12)]
    edgesl = np.array([[0, 1], [1, 0], [1, 1], [1, 5], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [5, 5], [5, 6], [4, 7], [6, 8],
             [7, 8], [4, 0], [3, 0], [5, 10], [10, 6], [6, 9], [9, 0], [7, 11]])
    weights = np.ones(21)
    g = sparse.csr_matrix((weights, (edgesl[:, 0], edgesl[:, 1])), shape=(12, 12))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    PR0 = nx.pagerank(G, alpha=0, max_iter=100, tol=1e-06, weight=None)
    print("Taxation parameter 0")
    print(PR0)
    pr = pagerank(g, p=0.0)
    print(pr)
    print("-------------------------------------------------------------")
    PR01 = nx.pagerank(G, alpha=0.1, max_iter=100, tol=1e-06, weight=None)
    print("Taxation parameter 0.1")
    print(PR01)
    pr = pagerank(g, p=0.1)
    print(pr)
    print("-------------------------------------------------------------")
    PR09 = nx.pagerank(G, alpha=0.9, max_iter=100, tol=1e-06, weight=None)
    print("Taxation parameter 0.9")
    print(PR09)
    pr = pagerank(g, p=0.9)
    print(pr)
    print("-------------------------------------------------------------")
    PR1 = nx.pagerank(G, alpha=1, max_iter=100, tol=1e-06, weight=None)
    print("Taxation parameter 1")
    print(PR1)
    pr = pagerank(g, p=1.0)
    print(pr)
    nx.draw(G, with_labels=True)
    plt.show()


def list4zad2():
    #https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html
    edges = [(1, 2), (2, 1), (2, 2), (2, 6), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1), (6, 6), (6, 7), (5, 8), (7, 9),
           (8, 9), (5, 1), (4, 1), (6, 11), (11, 7), (7, 10), (10, 1), (8, 12)]
    edgesl = np.array([[0, 1], [1, 0], [1, 1], [1, 5], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [5, 5], [5, 6], [4, 7], [6, 8],
             [7, 8], [4, 0], [3, 0], [5, 10], [10, 6], [6, 9], [9, 0], [7, 11]])
    weights = np.ones(21)
    g = sparse.csr_matrix((weights, (edgesl[:, 0], edgesl[:, 1])), shape=(12, 12))
    G = nx.DiGraph()
    G.add_nodes_from([1,2,3,4,5,6,7,8,9,10,11,12])
    G.add_edges_from(edges)
    PR0 = nx.pagerank(G, alpha=0, max_iter=100, tol=1e-06, weight=None, personalization = {6: 0.15, 8: 0.15})
    print("Taxation parameter 0")
    print(PR0)
    pr = pagerank(g, p=0.0)
    print(pr)
    print("-------------------------------------------------------------")
    PR01 = nx.pagerank(G, alpha=0.1, max_iter=100, tol=1e-06, weight=None, personalization = {6: 0.15, 8: 0.15})
    print("Taxation parameter 0.1")
    print(PR01)
    pr = pagerank(g, p=0.1)
    print(pr)
    print("-------------------------------------------------------------")
    PR09 = nx.pagerank(G, alpha=0.9, max_iter=100, tol=1e-06, weight=None, personalization = {6: 0.15, 8: 0.15})
    print("Taxation parameter 0.9")
    print(PR09)
    pr = pagerank(g, p=0.9)
    print(pr)
    print("-------------------------------------------------------------")
    PR1 = nx.pagerank(G, alpha=1, max_iter=100, tol=1e-06, weight=None, personalization = {6: 0.15, 8: 0.15})
    print("Taxation parameter 1")
    print(PR1)
    pr = pagerank(g, p=1.0)
    print(pr)
    nx.draw(G, with_labels=True)
    plt.show()

def list4zad3():
    #https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html
    edges = [(1, 2), (2, 1), (2, 2), (2, 6), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1), (6, 6), (6, 7), (5, 8), (7, 9),
           (8, 9), (5, 1), (4, 1), (6, 11), (11, 7), (7, 10), (10, 1), (8, 12)]
    edgesl = np.array([[0, 1], [1, 0], [1, 1], [1, 5], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [5, 5], [5, 6], [4, 7], [6, 8],
             [7, 8], [4, 0], [3, 0], [5, 10], [10, 6], [6, 9], [9, 0], [7, 11]])
    weights = np.ones(21)
    g = sparse.csr_matrix((weights, (edgesl[:, 0], edgesl[:, 1])), shape=(12, 12))
    G = nx.DiGraph()
    G.add_nodes_from([1,2,3,4,5,6,7,8,9,10,11,12])
    G.add_edges_from(edges)
    PR0 = nx.pagerank(G, alpha=0, max_iter=100, tol=1e-06, weight=None, personalization = {6: 0.15, 8: 0.15})
    print("Taxation parameter 0")
    print(PR0)
    pr = pagerank(g, p=0.0)
    print(pr)
    print("-------------------------------------------------------------")
    PR01 = nx.pagerank(G, alpha=0.1, max_iter=100, tol=1e-06, weight=None, personalization = {6: 0.15, 8: 0.15})
    print("Taxation parameter 0.1")
    print(PR01)
    pr = pagerank(g, p=0.1)
    print(pr)
    print("-------------------------------------------------------------")
    PR09 = nx.pagerank(G, alpha=0.9, max_iter=100, tol=1e-06, weight=None, personalization = {6: 0.15, 8: 0.15})
    print("Taxation parameter 0.9")
    print(PR09)
    pr = pagerank(g, p=0.9)
    print(pr)
    print("-------------------------------------------------------------")
    PR1 = nx.pagerank(G, alpha=1, max_iter=100, tol=1e-06, weight=None, personalization = {6: 0.15, 8: 0.15})
    print("Taxation parameter 1")
    print(PR1)
    pr = pagerank(g, p=1.0)
    print(pr)
    nx.draw(G, with_labels=True)
    plt.show()

if __name__ =="__main__":
    getTrustRank({1,2,6},12)