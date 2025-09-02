from numba import njit, jit
import numpy as np
from numpy import copy
import statistics
import fileinput
from plot import draw_perc_three_color


@njit
def find_perc(L, lattice):
    lattice = copy(lattice)
    lattice[0] = np.where(lattice[0] == 1, lattice[0] + 1, lattice[0])
    end = False
    for t in range(2, L ** 2 + 1):
        new = np.argwhere(lattice == t)
        if end:
            break
        if new.size != 0:
            for x, y in new:
                if x != L - 1:
                    if x == 0:
                        if 0 < y < (L - 1):
                            if lattice[x + 1][y] == 1:
                                lattice[x + 1][y] += t
                            if lattice[x][y + 1] == 1:
                                lattice[x][y + 1] += t

                            if lattice[x][y - 1] == 1:
                                lattice[x][y - 1] += t

                        elif y == 0:
                            if lattice[x + 1][y] == 1:
                                lattice[x + 1][y] += t

                            if lattice[x][y + 1] == 1:
                                lattice[x][y + 1] += t

                        else:
                            if lattice[x + 1][y] == 1:
                                lattice[x + 1][y] += t

                            if lattice[x][y - 1] == 1:
                                lattice[x][y - 1] += t

                    else:
                        if 0 < y < (L - 1):
                            if lattice[x + 1][y] == 1:
                                lattice[x + 1][y] += t

                            if lattice[x][y + 1] == 1:
                                lattice[x][y + 1] += t

                            if lattice[x][y - 1] == 1:
                                lattice[x][y - 1] += t

                            if lattice[x - 1][y] == 1:
                                lattice[x - 1][y] += t

                        elif y == 0:
                            if lattice[x + 1][y] == 1:
                                lattice[x + 1][y] += t

                            if lattice[x][y + 1] == 1:
                                lattice[x][y + 1] += t

                            if lattice[x - 1][y] == 1:
                                lattice[x - 1][y] += t

                        else:
                            if lattice[x + 1][y] == 1:
                                lattice[x + 1][y] += t

                            if lattice[x][y - 1] == 1:
                                lattice[x][y - 1] += t

                            if lattice[x - 1][y] == 1:
                                lattice[x - 1][y] += t
                else:
                    end = True
        else:
            break
    return end, lattice


@njit
def HK(L, lattice):
    lattice = copy(lattice)
    counter = 2
    for i in range(L):
        if np.count_nonzero(lattice[i] == 1) != 0:
            for j in range(L):
                if lattice[i][j] == 1:
                    if i == 0 and j != 0:
                        if lattice[i][j - 1] != 0:
                            lattice[i][j] = lattice[i][j - 1]
                        else:
                            lattice[i][j] = counter
                            counter += 1
                    elif i != 0 and j == 0:
                        if lattice[i - 1][j] != 0:
                            lattice[i][j] = lattice[i - 1][j]
                        else:
                            lattice[i][j] = counter
                            counter += 1
                    elif i != 0 and j != 0:
                        if lattice[i - 1][j] != 0 and lattice[i][j - 1] != 0:
                            if lattice[i - 1][j] > lattice[i][j - 1]:
                                lattice[i][j] = lattice[i][j - 1]
                                new = np.argwhere(lattice == lattice[i - 1][j])
                                for x, y in new:
                                    lattice[x][y] = lattice[i][j - 1]
                            elif lattice[i - 1][j] < lattice[i][j - 1]:
                                lattice[i][j] = lattice[i - 1][j]
                                new = np.argwhere(lattice == lattice[i][j - 1])
                                for x, y in new:
                                    lattice[x][y] = lattice[i - 1][j]
                            else:
                                lattice[i][j] = lattice[i][j - 1]
                        elif lattice[i - 1][j] == 0 and lattice[i][j - 1] != 0:
                            lattice[i][j] = lattice[i][j - 1]
                        elif lattice[i - 1][j] != 0 and lattice[i][j - 1] == 0:
                            lattice[i][j] = lattice[i - 1][j]
                        else:
                            lattice[i][j] = counter
                            counter += 1
                    else:
                        lattice[i][j] = counter
                        counter += 1
    return lattice


@jit
def p_critical(L, T):
    L = L
    P = np.linspace(0, 1, 101)
    T = T
    prob = []
    for p in P:
        print(p)
        is_ended = []
        for t in range(T):
            lattice = np.random.choice(2, [L, L], p=[1 - p, p])
            lattice[0] = np.where(lattice[0] == 1, lattice[0] + 1, lattice[0])
            end, perc_lattice = find_perc(L, lattice)
            is_ended.append(end)
        prob.append(is_ended.count(True))
    return P, prob


@jit
def distribution(L, T, p0, dp, pk):
    L = L
    number_of_steps = int((pk - p0) / dp) + 1
    P = np.linspace(p0, pk, number_of_steps)
    P = np.append(P, np.array([0.592746]))
    T = T
    prob = []
    for p in P:
        print(p)
        dist = dict()
        for t in range(T):
            lattice = np.random.choice(2, [L, L], p=[1 - p, p])
            clusters = HK(L, lattice)
            max_size, array_of_clusters = max_cluster_size(clusters)
            nspL = dict()
            if len(array_of_clusters) != 0:
                array_of_clusters_sizes = checking_cluster_sizes(clusters, array_of_clusters)
                for s in array_of_clusters_sizes:
                    if s in nspL.keys():
                        nspL[s] += (1 / len(array_of_clusters_sizes))
                    else:
                        nspL[s] = (1 / len(array_of_clusters_sizes))
                for k in nspL.keys():
                    if k in dist.keys():
                        dist[k] += nspL[k] / T
                    else:
                        dist[k] = nspL[k] / T
        dist_of_clusters_to_file(dist, p, L, T)
    return P, prob


@jit
def p_critical(L, T, p0, dp, pk):
    L = L
    number_of_steps = int((pk - p0) / dp) + 1
    P = np.linspace(p0, pk, number_of_steps)
    P = np.append(P, np.array([0.592746]))
    T = T
    prob = []
    avg_cluster_sizes = []
    for p in P:
        print(p)
        is_ended = []
        max_cluster_sizes = []
        for t in range(T):
            lattice = np.random.choice(2, [L, L], p=[1 - p, p])
            lattice[0] = np.where(lattice[0] == 1, lattice[0] + 1, lattice[0])
            end, perc_lattice = find_perc(L, lattice)
            clusters = HK(L, lattice)
            max_size, array_of_clusters = max_cluster_size(clusters)
            is_ended.append(end)
            max_cluster_sizes.append(max_size)
        prob.append(is_ended.count(True))
        avg_cluster_sizes.append(statistics.fmean(max_cluster_sizes))
    prob = list(np.array(prob) / T)
    p_critical_to_file(P, prob, avg_cluster_sizes, L, T)
    return P, prob


def p_critical_to_file(P, p_flow, avg_cluster_sizes, L, T):
    with open("output_data\Ave_L" + str(L) + "T" + str(T) + ".txt", 'w') as f:
        for p, p_f, avg in zip(P, p_flow, avg_cluster_sizes):
            f.write(str(p) + "  " + str(p_f) + "  " + str(avg) + "\n")
        f.close()


def dist_of_clusters_to_file(nspL, p, L, T):
    print(str(format(round(p, 3), '.2f')))
    with open("output_data/Dist_p" + str(format(round(p, 3), '.2f')) + "L" + str(L) + "T" + str(T) + ".txt", 'w') as f:
        for s, n in zip(nspL.keys(), nspL.values()):
            f.write(str(s) + "  " + str(float(n)) + "\n")
        f.close()


@njit
def max_cluster_size(lattice):
    lattice = copy(lattice)
    min_cluster_number = 2
    max_cluster_number = np.max(lattice)
    clusters_in_lattice = []
    max_size = 0
    for size in range(min_cluster_number, max_cluster_number + 1):
        array_of_indexes = np.argwhere(lattice == size)
        size_of_cluster = len(array_of_indexes)
        if size_of_cluster != 0:
            clusters_in_lattice.append(size)
            if size_of_cluster > max_size:
                max_size = size_of_cluster
    return max_size, clusters_in_lattice


@njit
def normalize_clusters(lattice, clusters_in_lattice):
    lattice = copy(lattice)
    start = 1
    new_clusters_in_lattice = [0]
    for cil in clusters_in_lattice:
        array_of_indexes = np.argwhere(lattice == cil)
        new_clusters_in_lattice.append(start)
        for x, y in array_of_indexes:
            lattice[x][y] = start
        start += 1
    return lattice, new_clusters_in_lattice


@njit
def checking_cluster_sizes(clusters, array_of_clusters):
    clusters, array_of_clusters = clusters, array_of_clusters
    array_of_clusters_sizes = []
    for cluster in array_of_clusters:
        indexes = np.argwhere(clusters == cluster)
        array_of_clusters_sizes.append(len(indexes))
    return array_of_clusters_sizes


if __name__ == "__main__":
    data_from_file = dict()
    for line in fileinput.input():
        datal = line.replace("\n", "").replace(" ", "").split(sep="%")
        data_from_file[datal[1]] = datal[0]
    print(data_from_file)

    L = int(data_from_file['L'])
    T = int(data_from_file['T'])
    p0 = float(data_from_file['p_0'])
    pk = float(data_from_file['p_k'])
    dp = float(data_from_file['dp'])
    number_of_steps = int((pk - p0) / dp) + 1
    P = np.linspace(p0, pk, number_of_steps)
    distribution(L=L, T=T, p0=p0, pk=pk, dp=dp)
    p_critical(L=L, T=T, p0=p0, pk=pk, dp=dp)
    draw_perc_three_color(perc_lattice, L)
