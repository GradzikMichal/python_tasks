import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import matrix_power
from numba import njit
from numba.core import types
from numba.typed import Dict
import numba


def rfile(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            (node, edge) = line.rstrip().split(' ')
            node, edge = int(node), int(edge)
            data.append((node, edge))
    f.close()
    return data


@njit
def numb_of_links(data):
    node_edge_number = Dict.empty(
        key_type=types.int64,
        value_type=types.int64
    )
    for node_edge in data:
        keys = []
        for i in node_edge_number.keys():
            keys.append(i)
        if node_edge[0] in keys:
            node_edge_number[node_edge[0]] += 1
        else:
            keys.append(node_edge[0])
            node_edge_number[node_edge[0]] = types.int64(1)
        if node_edge[1] in keys:
            node_edge_number[node_edge[1]] += 1
        else:
            keys.append(node_edge[1])
            node_edge_number[node_edge[1]] = types.int64(1)
    return node_edge_number


@njit
def degree_dist(node_edge_number, N):
    degree_distrib = Dict.empty(
        key_type=types.int64,
        value_type=types.int64
    )
    for n in range(N):
        keys = []
        for i in node_edge_number.keys():
            keys.append(i)
        if n not in keys:
            node_edge_number[n] = 0
    d_dist = Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    number_of_nodes = max(node_edge_number) + 1
    for k, v in node_edge_number.items():
        keys = []
        for i in degree_distrib.keys():
            keys.append(i)
        if v in keys:
            degree_distrib[v] += 1
        else:
            degree_distrib[v] = types.int64(1)
    for k, v in degree_distrib.items():
        d_dist[k] = types.float64(v / number_of_nodes)
    return d_dist


@njit
def average_degree(node_edge_number):
    values = []
    for i in node_edge_number.values():
        values.append(i)
    degree_sum = sum(values)  # 2 x poniewaz w danych nie ma 2-1 jezeli jest 1-2
    number_of_nodes = len(node_edge_number.keys())  # from https://snap.stanford.edu/data/ego-Facebook.html
    L = 0
    for _, v in node_edge_number.items():
        L += v
    # avg_data = (2 * L) / number_of_nodes
    avg_degree = degree_sum / number_of_nodes
    return avg_degree


@njit(parallel=True)
def data_to_array(data):
    max_node = 4039
    array = np.zeros((max_node, max_node))
    for node1, node2 in data:
        array[node1][node2] = 1
        array[node2][node1] = 1
    return array


@njit(parallel=True)
def clusterCoeff(matrix, node_edge_number):
    matrix_pow_3 = matrix_power(matrix, 3)
    num_nodes = len(matrix)
    c_i = np.zeros(num_nodes, dtype=np.float64)
    for i in range(0, num_nodes):
        keys = []
        for j in node_edge_number.keys():
            keys.append(j)
        if i in keys:
            if node_edge_number[i] > 1:
                c = matrix_pow_3[i][i] / (float(node_edge_number[i]) * (float(node_edge_number[i]) - 1))
            else:
                c = 0.0
        else:
            c = 0.0
        c_i[i] = c
    c_avg = np.sum(c_i) / num_nodes
    return c_avg, c_i


@njit(parallel=True)
def dist_of_shortest_path_from_node(n, aggregated_nodes, max=4039):
    aggregated_nodes = aggregated_nodes
    unvisited_nodes = np.zeros(max)
    distance = np.ones(max) * np.inf
    current_node = n
    distance[n] = 0
    end = 0
    while sum(unvisited_nodes) != max:
        min_distance_possible = 1
        neighbors_of_current_node = aggregated_nodes[current_node]
        for nn in neighbors_of_current_node:
            if unvisited_nodes[nn] == 0:
                if distance[nn] > distance[current_node] + 1:
                    distance[nn] = distance[current_node] + 1
        unvisited_nodes[current_node] = 1
        new_current_node = 0
        while True:
            possible_current_nodes = np.argwhere(distance == min_distance_possible)
            for possible_current in possible_current_nodes:
                if unvisited_nodes[possible_current[0]] == 0:
                    current_node = possible_current[0]
                    new_current_node = 1
                    break
            if new_current_node == 1:
                break
            if len(np.argwhere(distance == min_distance_possible + 1)) == 0:
                end = 1
                break
            min_distance_possible += 1
        if end:
            break
    return distance


@njit(nogil=True, parallel=True)
def dijkstra_algorithm(agr_data):
    dist_matrix = []
    for node in agr_data.keys():
        distances_from_node = dist_of_shortest_path_from_node(node, aggregated_nodes=agr_data)
        dist_matrix.append(distances_from_node)
    return dist_matrix


@njit(parallel=True)
def aggregate_neighbors(data):
    aggregate = Dict.empty(
        key_type=types.int64,
        value_type=types.ListType(types.int64[:])
    )
    for n1, n2 in data:
        keys = []
        for i in aggregate.keys():
            keys.append(i)
        if n1 in keys:
            aggregate[n1].append(n2)
        else:
            aggregate[types.int64(n1)] = types.ListType([types.int64(n2)])
        if n2 in keys:
            aggregate[n2].append(n1)
        else:
            aggregate[types.int64(n2)] = types.ListType([types.int64(n1)])
    return aggregate


@njit(parallel=True)
def floyd_warshall(data, maks=4039):
    distance = np.ones((maks, maks)) * np.inf
    for (u, v) in data:
        distance[u][v] = 1
        distance[v][u] = 1
    for i in range(0, maks):
        distance[i][i] = 0
    for k in range(0, maks):
        for i in range(0, maks):
            for j in range(0, maks):
                if distance[i][j] > distance[i][k] + distance[k][j]:
                    distance[i][j] = distance[i][k] + distance[k][j]
    return distance


@njit(parallel=True)
def shortest_path_diameter_avg_path(paths):
    data = paths
    data = data.flatten()
    data = sorted(list(set(data)))

    if np.inf in data:
        conv = []
        for d in data[:-1]:
            conv.append(int(d))
        diameter = int(max(conv))
    else:
        diameter = int(max(data))
    dist_of_paths = Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    sum = 0
    for i in range(diameter + 1):

        number_of_path = np.argwhere(paths == i)
        value = len(number_of_path)
        if i != 0:
            value = float(value) / 2
        dist_of_paths[i] = types.float64(value)
        sum += value
    sum += len(np.argwhere(paths == np.inf)) / 2
    avg_path = 0.0
    for i in dist_of_paths.keys():
        dist_of_paths[i] = types.float64(dist_of_paths[i] / sum)
        avg_path += i * dist_of_paths[i]
    return dist_of_paths, diameter, avg_path


@njit
def erdos_renyi_model(N, L):
    adjacent_array = np.zeros((N, N))
    while len(np.argwhere(adjacent_array != 0)) != L:
        first_node = np.random.randint(0, N)
        second_node = np.random.randint(0, N)
        if first_node != second_node and adjacent_array[first_node][second_node] == 0 and adjacent_array[second_node][
            first_node] == 0:
            adjacent_array[first_node][second_node] = 1
    edge_list = np.argwhere(adjacent_array != 0)
    return adjacent_array, edge_list


@njit
def erdos_renyi_gilbert_model(N, p):
    adjacent_array = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if p >= np.random.rand(1)[0]:
                adjacent_array[i][j] = 1
    edge_list = np.argwhere(adjacent_array != 0)
    for i in range(N):
        for j in range(i + 1, N):
            if adjacent_array[i][j] == 1:
                adjacent_array[j][i] = 1
    return adjacent_array, edge_list


@njit
def watts_strogatz_model(N, k, beta):
    edge_list = set()
    number_of_neighbors_in_one_direction = int(k / 2)
    for i in range(N):
        for nn in range(1, number_of_neighbors_in_one_direction + 1):
            edge1 = i - nn
            edge2 = i + nn
            if edge1 < 0:
                edge1 = N + edge1
            if edge2 > N - 1:
                edge2 = edge2 - N
            if (i, edge1) not in edge_list and (edge1, i) not in edge_list:
                edge_list.add((i, edge1))
            if (i, edge2) not in edge_list and (edge2, i) not in edge_list:
                edge_list.add((i, edge2))
    edge_list = list(edge_list)
    new_edge_list = []
    for edge in edge_list:
        if beta >= np.random.rand(1)[0]:
            new_node = np.random.randint(0, N)
            new_node2 = np.random.randint(0, N)
            while new_node == edge[0] or new_node == edge[1] and (edge[0], new_node) in new_edge_list and (
            new_node, edge[0]) in new_edge_list:
                new_node = np.random.randint(0, N)
            new_edge_list.append((edge[0], new_node))
            while new_node2 == edge[0] or new_node2 == edge[1] and (edge[1], new_node2) in new_edge_list and (
            new_node2, edge[1]) in new_edge_list:
                new_node2 = np.random.randint(0, N)
            new_edge_list.append((new_node2, edge[1]))
        else:
            new_edge_list.append(edge)

    adjacent_array = np.zeros((N, N))

    for (node1, node2) in new_edge_list:
        adjacent_array[node1][node2] = 1
        adjacent_array[node2][node1] = 1
    return adjacent_array, new_edge_list


@njit
def erdos_renyi_graphs_data_gen(N_list, L_list, T):
    N_list = N_list
    L_list = L_list
    T = T
    dist_list = []
    avg_list = []
    avg_c_avg_list = []
    avg_c_i_list = []
    avg_dist_of_paths_list = []
    avg_diameter_list = []
    avg_path_list = []
    for N, L in zip(N_list, L_list):
        avg_avg_degree = 0.0
        avg_c_avg = 0.0
        avg_c_i = np.zeros(N, dtype=np.float64)
        avg_distribution = Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        avg_diameter = 0.0
        avg_dist_of_paths = Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        avg_avg_path = 0.0
        for _ in range(T):
            adjacent_array, edge_list = erdos_renyi_model(N, L)
            node_edge_number = numb_of_links(edge_list)
            distribution = degree_dist(node_edge_number, N)
            avg_degree = average_degree(node_edge_number)
            avg_avg_degree += avg_degree
            c_avg, c_i = clusterCoeff(adjacent_array, node_edge_number=node_edge_number)
            avg_c_avg += c_avg
            avg_c_i += c_i
            keys = []

            for k, v in distribution.items():
                for i in avg_distribution.keys():
                    keys.append(i)
                if k in keys:
                    avg_distribution[k] += v
                else:
                    avg_distribution[k] = types.float64(v)
            distances_from_node = floyd_warshall(edge_list, N)
            dist_of_paths, diameter, avg_path = shortest_path_diameter_avg_path(distances_from_node)
            avg_avg_path += avg_path
            keys = []

            for k, v in dist_of_paths.items():
                for i in avg_dist_of_paths.keys():
                    keys.append(i)
                if k in keys:
                    avg_dist_of_paths[k] += v
                else:
                    avg_dist_of_paths[k] = types.float64(v)
            avg_diameter += diameter
        dist_list.append(avg_distribution)
        avg_list.append(avg_avg_degree / T)
        avg_c_avg_list.append(avg_c_avg / N)
        avg_c_i_list.append(avg_c_i)
        avg_dist_of_paths_list.append(avg_dist_of_paths)
        avg_diameter_list.append(avg_diameter / T)
        avg_path_list.append(avg_avg_path / T)
    return dist_list, avg_list, avg_c_avg_list, avg_c_i_list, avg_dist_of_paths_list, avg_diameter_list, avg_path_list


def erdos_renyi_graphs_data_plot(N_list, L_list, T):
    N_list = N_list
    L_list = L_list
    T = T
    dist_list, avg_list, avg_c_avg_list, avg_c_i_list, avg_dist_of_paths_list, avg_diameter_list, avg_path_list = erdos_renyi_graphs_data_gen(
        N_list, L_list, T)
    labels = []
    graph = []
    plt.grid(linestyle='dotted', linewidth=1)
    for dist, n, l, a in zip(dist_list, N_list, L_list, avg_list):
        labels.append("n=" + str(n) + " l=" + str(l) + " avg=" + str(round(a, 3)))
        g = plt.bar(dist.keys(), np.array(list(dist.values())) / T, alpha=0.6)
        graph.append(g)
    plt.legend(graph, labels=labels, loc="upper right")
    plt.xlim([-1, 80])
    plt.xticks(range(-1, 80))
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.show()

    plt.scatter(L_list, avg_c_avg_list)
    plt.xlabel("Number of edges")
    plt.ylabel("Average cluster coefficient")
    plt.grid(linestyle='dotted', linewidth=1)
    plt.show()

    graph1 = []
    bins = np.linspace(0, 1, 21)
    labels = []
    for b in range(len(bins) - 1):
        labels.append(str(round(bins[b], 3)) + " - " + str(round(bins[b + 1], 3)))
    mv = [-0.01, 0.0, 0.01, 0.02, 0.03]
    labelsl = []
    for c_i, m_v, n, l in zip(avg_c_i_list, mv, N_list, L_list):
        c_i = list(np.array(c_i) / n)
        labelsl.append("n=" + str(n) + " l=" + str(l))
        whereBelong = np.digitize(x=c_i, bins=bins[1:], right=True)
        counts = np.zeros(len(bins[1:]))
        for i in whereBelong:
            counts[i] += 1
        cent_bins = []
        for i in range(len(bins) - 1):
            cent_bins.append(np.mean([bins[i], bins[i + 1]]))
        g = plt.bar(np.array(cent_bins) + m_v, counts / (n), alpha=0.6, width=0.01)
        graph1.append(g)
    plt.legend(graph1, labels=labelsl, loc="upper right")
    plt.ylabel("Number of Occurrences")
    plt.xticks(bins[:-4] + 0.035, labels[:-3])
    plt.xlabel("Cluster Coefficient")
    plt.grid(linestyle='dotted', linewidth=1)
    plt.xlim([-0.05, 0.85])
    plt.show()

    labels = []
    plt.grid(linestyle='dotted', linewidth=1)
    mv = [-0.2, -0.1, 0.0, 0.1, 0.2]
    for dist, n, l, a, m_v in zip(avg_dist_of_paths_list, N_list, L_list, avg_path_list, mv):
        labels.append("n=" + str(n) + " l=" + str(l) + " avg=" + str(round(a, 3)))
        g = plt.bar(np.array(list(dist.keys())) + m_v, np.array(list(dist.values())) / T, alpha=0.6, width=0.1)
        graph.append(g)
    plt.legend(graph, labels=labels, loc="upper right")
    plt.xlim([-1, 11])
    plt.xticks(range(-1, 11))
    plt.xlabel("Length of path")
    plt.ylabel("Distribution of path length")
    plt.show()

    plt.scatter(L_list, avg_diameter_list)
    plt.xlabel("Number of edges")
    plt.ylabel("Average diameter")
    plt.grid(linestyle='dotted', linewidth=1)
    plt.show()


@njit
def erdos_renyi_gilbert_graphs_data_gen(N_list, p_list, T):
    N_list = N_list
    p_list = p_list
    T = T
    dist_list = []
    avg_list = []
    avg_c_avg_list = []
    avg_c_i_list = []
    avg_dist_of_paths_list = []
    avg_diameter_list = []
    avg_path_list = []
    for N, L in zip(N_list, p_list):
        avg_avg_degree = 0.0
        avg_c_avg = 0.0
        avg_c_i = np.zeros(N, dtype=np.float64)
        avg_distribution = Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        avg_diameter = 0.0
        avg_dist_of_paths = Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        avg_avg_path = 0.0
        for _ in range(T):
            adjacent_array, edge_list = erdos_renyi_gilbert_model(N, L)
            node_edge_number = numb_of_links(edge_list)
            distribution = degree_dist(node_edge_number, N)
            avg_degree = average_degree(node_edge_number)
            avg_avg_degree += avg_degree
            c_avg, c_i = clusterCoeff(adjacent_array, node_edge_number=node_edge_number)
            avg_c_avg += c_avg
            avg_c_i += c_i
            keys = []

            for k, v in distribution.items():
                for i in avg_distribution.keys():
                    keys.append(i)
                if k in keys:
                    avg_distribution[k] += v
                else:
                    avg_distribution[k] = types.float64(v)
            distances_from_node = floyd_warshall(edge_list, N)
            dist_of_paths, diameter, avg_path = shortest_path_diameter_avg_path(distances_from_node)
            avg_avg_path += avg_path
            keys = []

            for k, v in dist_of_paths.items():
                for i in avg_dist_of_paths.keys():
                    keys.append(i)
                if k in keys:
                    avg_dist_of_paths[k] += v
                else:
                    avg_dist_of_paths[k] = types.float64(v)
            avg_diameter += diameter
        dist_list.append(avg_distribution)
        avg_list.append(avg_avg_degree / T)
        avg_c_avg_list.append(avg_c_avg / T)
        avg_c_i_list.append(avg_c_i)
        avg_dist_of_paths_list.append(avg_dist_of_paths)
        avg_diameter_list.append(avg_diameter / T)
        avg_path_list.append(avg_avg_path / T)
    return dist_list, avg_list, avg_c_avg_list, avg_c_i_list, avg_dist_of_paths_list, avg_diameter_list, avg_path_list


def erdos_renyi_gilbert_graphs_data_plot(N_list, p_list, T):
    N_list = N_list
    p_list = p_list
    T = T
    dist_list, avg_list, avg_c_avg_list, avg_c_i_list, avg_dist_of_paths_list, avg_diameter_list, avg_path_list = erdos_renyi_gilbert_graphs_data_gen(
        N_list, p_list, T)
    labels = []
    graph = []
    plt.grid(linestyle='dotted', linewidth=1)
    for dist, n, p, a in zip(dist_list, N_list, p_list, avg_list):
        labels.append("n=" + str(n) + " p=" + str(p) + " avg=" + str(round(a, 3)))
        g = plt.bar(dist.keys(), np.array(list(dist.values())) / T, alpha=0.6)
        graph.append(g)
    plt.legend(graph, labels=labels, loc="upper right")
    plt.xlim([0, 100])
    plt.xticks(range(0, 100, 5))
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.show()

    plt.scatter(p_list, avg_c_avg_list)
    plt.xlabel("Probability of edge between nodes")
    plt.ylabel("Average cluster coefficient")
    plt.grid(linestyle='dotted', linewidth=1)
    plt.show()

    graph1 = []
    bins = np.linspace(0, 1, 21)
    labels = []
    for b in range(len(bins) - 1):
        labels.append(str(round(bins[b], 3)) + " - " + str(round(bins[b + 1], 3)))
    mv = [-0.01, 0.0, 0.01, 0.02, 0.03]
    labelsl = []
    for c_i, m_v, n, p in zip(avg_c_i_list, mv, N_list, p_list):
        c_i = list(np.array(c_i) / T)
        labelsl.append("n=" + str(n) + " p=" + str(p))
        whereBelong = np.digitize(x=c_i, bins=bins[1:], right=True)
        counts = np.zeros(len(bins[1:]))
        for i in whereBelong:
            counts[i] += 1
        cent_bins = []

        for i in range(len(bins) - 1):
            cent_bins.append(np.mean([bins[i], bins[i + 1]]))
        g = plt.bar(np.array(cent_bins) + m_v, counts / (n), alpha=0.6, width=0.01)
        graph1.append(g)
    plt.legend(graph1, labels=labelsl, loc="upper right")
    plt.ylabel("Number of Occurrences")
    plt.xticks(bins[:-1] + 0.035, labels)
    plt.xlabel("Cluster Coefficient")
    plt.grid(linestyle='dotted', linewidth=1)
    plt.xlim([0.0, 1.])
    plt.show()

    labels = []
    plt.grid(linestyle='dotted', linewidth=1)
    mv = [-0.2, -0.1, 0.0, 0.1, 0.2]
    for dist, n, p, a, m_v in zip(avg_dist_of_paths_list, N_list, p_list, avg_path_list, mv):
        labels.append("n=" + str(n) + " p=" + str(p) + " avg=" + str(round(a, 3)))
        g = plt.bar(np.array(list(dist.keys())) + m_v, np.array(list(dist.values())) / T, alpha=0.6, width=0.1)
        graph.append(g)
    plt.legend(graph, labels=labels, loc="upper right")
    plt.xlim([-1, 11])
    plt.xticks(range(-1, 11))
    plt.xlabel("Length of path", fontsize=16)
    plt.ylabel("Distribution of path length", fontsize=16)
    plt.show()

    plt.scatter(p_list, avg_diameter_list)
    plt.xlabel("Probability p of edge", fontsize=16)
    plt.ylabel("Average diameter", fontsize=16)
    plt.grid(linestyle='dotted', linewidth=1)
    plt.show()


@njit
def watts_strogatz_graphs_data_gen(N_list, k_list, beta_list, T):
    N_list = N_list
    k_list = k_list
    beta_list = beta_list
    T = T
    dist_list = []
    avg_list = []
    avg_c_avg_list = []
    avg_c_i_list = []
    avg_dist_of_paths_list = []
    avg_diameter_list = []
    avg_path_list = []
    for N, K, beta in zip(N_list, k_list, beta_list):
        avg_avg_degree = 0.0
        avg_c_avg = 0.0
        avg_c_i = np.zeros(N, dtype=np.float64)
        avg_distribution = Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        avg_diameter = 0.0
        avg_dist_of_paths = Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        avg_avg_path = 0.0
        for _ in range(T):
            adjacent_array, edge_list = watts_strogatz_model(N, K, beta)
            node_edge_number = numb_of_links(edge_list)
            distribution = degree_dist(node_edge_number, N)
            avg_degree = average_degree(node_edge_number)
            avg_avg_degree += avg_degree
            c_avg, c_i = clusterCoeff(adjacent_array, node_edge_number=node_edge_number)
            avg_c_avg += c_avg
            avg_c_i += c_i
            keys = []

            for k, v in distribution.items():
                for i in avg_distribution.keys():
                    keys.append(i)
                if k in keys:
                    avg_distribution[k] += v
                else:
                    avg_distribution[k] = types.float64(v)
            distances_from_node = floyd_warshall(edge_list, N)
            dist_of_paths, diameter, avg_path = shortest_path_diameter_avg_path(distances_from_node)
            avg_avg_path += avg_path
            keys = []

            for k, v in dist_of_paths.items():
                for i in avg_dist_of_paths.keys():
                    keys.append(i)
                if k in keys:
                    avg_dist_of_paths[k] += v
                else:
                    avg_dist_of_paths[k] = types.float64(v)
            avg_diameter += diameter
        dist_list.append(avg_distribution)
        avg_list.append(avg_avg_degree / T)
        avg_c_avg_list.append(avg_c_avg / T)
        avg_c_i_list.append(avg_c_i)
        avg_dist_of_paths_list.append(avg_dist_of_paths)
        avg_diameter_list.append(avg_diameter / T)
        avg_path_list.append(avg_avg_path / T)
    return dist_list, avg_list, avg_c_avg_list, avg_c_i_list, avg_dist_of_paths_list, avg_diameter_list, avg_path_list


def watts_strogatz_graphs_plots(N_list, k_list, beta_list, T):
    N_list = N_list
    k_list = k_list
    beta_list = beta_list
    T = T
    dist_list, avg_list, avg_c_avg_list, avg_c_i_list, avg_dist_of_paths_list, avg_diameter_list, avg_path_list = watts_strogatz_graphs_data_gen(
        N_list, k_list, beta_list, T)
    labels = []
    graph = []
    plt.grid(linestyle='dotted', linewidth=1)
    for dist, n, k, b, a in zip(dist_list, N_list, k_list, beta_list, avg_list):
        labels.append("n=" + str(n) + " k=" + str(k) + " beta=" + str(b) + " avg=" + str(round(a, 3)))
        g = plt.bar(dist.keys(), np.array(list(dist.values())) / T, alpha=0.6)
        graph.append(g)
    plt.legend(graph, labels=labels, loc="upper right")
    #plt.xlim([0, 100])
    #plt.xticks(range(0, 100, 5))
    plt.xlabel("Degree of node k")
    plt.ylabel("P(k)")
    plt.show()

    plt.scatter(beta_list, avg_c_avg_list)
    plt.xlabel("Probability p of edge")
    plt.ylabel("Average cluster coefficient")
    plt.grid(linestyle='dotted', linewidth=1)
    plt.show()

    graph1 = []
    bins = np.linspace(0, 1, 21)
    labels = []
    for b in range(len(bins) - 1):
        labels.append(str(round(bins[b], 3)) + " - " + str(round(bins[b + 1], 3)))
    mv = [0, 0.01, 0.02, 0.03, 0.04]
    labelsl = []
    for c_i, m_v, n, k, b in zip(avg_c_i_list, mv, N_list, k_list, beta_list):
        c_i = list(np.array(c_i) / T)
        labelsl.append("n=" + str(n) + " k=" + str(k) + " beta=" + str(b))
        whereBelong = np.digitize(x=c_i, bins=bins[1:], right=True)
        counts = np.zeros(len(bins[1:]))
        for i in whereBelong:
            counts[i] += 1
        cent_bins = []

        for i in range(len(bins) - 1):
            cent_bins.append(np.mean([bins[i], bins[i + 1]]))
        g = plt.bar(np.array(cent_bins) + m_v, counts / (n), alpha=0.6, width=0.01)
        graph1.append(g)
    plt.legend(graph1, labels=labelsl, loc="upper right")
    plt.ylabel("Number of Occurrences")
    plt.xticks(bins[:-1] + 0.035, labels)
    plt.xlabel("Cluster Coefficient")
    plt.grid(linestyle='dotted', linewidth=1)
    plt.xlim([0.0, 1.])
    plt.show()

    labels = []
    plt.grid(linestyle='dotted', linewidth=1)
    mv = [-0.2, -0.1, 0.0, 0.1, 0.2]
    for dist, n, k, a, m_v, b in zip(avg_dist_of_paths_list, N_list, k_list, avg_path_list, mv, beta_list):
        labels.append("n=" + str(n) + " k=" + str(k) + " beta=" + str(b) + " avg=" + str(round(a, 3)))
        g = plt.bar(np.array(list(dist.keys())) + m_v, np.array(list(dist.values())) / T, alpha=0.6, width=0.1)
        graph.append(g)
    plt.legend(graph, labels=labels, loc="upper right")
    plt.xlim([-1, 11])
    plt.xticks(range(-1, 11))
    plt.xlabel("Length of path")
    plt.ylabel("Distribution of path length")
    plt.show()

    plt.scatter(k_list, avg_diameter_list)
    plt.xlabel("Number of edges k")
    plt.ylabel("Average diameter")
    plt.grid(linestyle='dotted', linewidth=1)
    plt.show()


@njit
def check_all_models(N, K, beta, p, T, L):
    N = N
    L = L
    K = K
    beta = beta
    T = T
    p = p
    dist_list = []
    avg_list = []
    avg_c_avg_list = []
    avg_c_i_list = []
    avg_dist_of_paths_list = []
    avg_diameter_list = []
    avg_path_list = []
    ############## watts
    avg_avg_degree = 0.0
    avg_c_avg = 0.0
    avg_c_i = np.zeros(N, dtype=np.float64)
    avg_distribution = Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    avg_diameter = 0.0
    avg_dist_of_paths = Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    avg_avg_path = 0.0
    for _ in range(T):
        adjacent_array, edge_list = watts_strogatz_model(N, K, beta)
        node_edge_number = numb_of_links(edge_list)
        distribution = degree_dist(node_edge_number, N)
        avg_degree = average_degree(node_edge_number)
        avg_avg_degree += avg_degree
        c_avg, c_i = clusterCoeff(adjacent_array, node_edge_number=node_edge_number)
        avg_c_avg += c_avg
        avg_c_i += c_i
        keys = []

        for k, v in distribution.items():
            for i in avg_distribution.keys():
                keys.append(i)
            if k in keys:
                avg_distribution[k] += v
            else:
                avg_distribution[k] = types.float64(v)
        distances_from_node = floyd_warshall(edge_list, N)
        dist_of_paths, diameter, avg_path = shortest_path_diameter_avg_path(distances_from_node)
        avg_avg_path += avg_path
        keys = []

        for k, v in dist_of_paths.items():
            for i in avg_dist_of_paths.keys():
                keys.append(i)
            if k in keys:
                avg_dist_of_paths[k] += v
            else:
                avg_dist_of_paths[k] = types.float64(v)
        avg_diameter += diameter
    dist_list.append(avg_distribution)
    avg_list.append(avg_avg_degree / T)
    avg_c_avg_list.append(avg_c_avg / T)
    avg_c_i_list.append(avg_c_i)
    avg_dist_of_paths_list.append(avg_dist_of_paths)
    avg_diameter_list.append(avg_diameter / T)
    avg_path_list.append(avg_avg_path / T)
    ################### gilbert
    avg_avg_degree = 0.0
    avg_c_avg = 0.0
    avg_c_i = np.zeros(N, dtype=np.float64)
    avg_distribution = Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    avg_diameter = 0.0
    avg_dist_of_paths = Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    avg_avg_path = 0.0
    for _ in range(T):
        adjacent_array, edge_list = erdos_renyi_gilbert_model(N, p)
        node_edge_number = numb_of_links(edge_list)
        distribution = degree_dist(node_edge_number, N)
        avg_degree = average_degree(node_edge_number)
        avg_avg_degree += avg_degree
        c_avg, c_i = clusterCoeff(adjacent_array, node_edge_number=node_edge_number)
        avg_c_avg += c_avg
        avg_c_i += c_i
        keys = []

        for k, v in distribution.items():
            for i in avg_distribution.keys():
                keys.append(i)
            if k in keys:
                avg_distribution[k] += v
            else:
                avg_distribution[k] = types.float64(v)
        distances_from_node = floyd_warshall(edge_list, N)
        dist_of_paths, diameter, avg_path = shortest_path_diameter_avg_path(distances_from_node)
        avg_avg_path += avg_path
        keys = []

        for k, v in dist_of_paths.items():
            for i in avg_dist_of_paths.keys():
                keys.append(i)
            if k in keys:
                avg_dist_of_paths[k] += v
            else:
                avg_dist_of_paths[k] = types.float64(v)
        avg_diameter += diameter
    dist_list.append(avg_distribution)
    avg_list.append(avg_avg_degree / T)
    avg_c_avg_list.append(avg_c_avg / T)
    avg_c_i_list.append(avg_c_i)
    avg_dist_of_paths_list.append(avg_dist_of_paths)
    avg_diameter_list.append(avg_diameter / T)
    avg_path_list.append(avg_avg_path / T)
    #############################Erdos-renei
    avg_avg_degree = 0.0
    avg_c_avg = 0.0
    avg_c_i = np.zeros(N, dtype=np.float64)
    avg_distribution = Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    avg_diameter = 0.0
    avg_dist_of_paths = Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    avg_avg_path = 0.0
    for _ in range(T):
        adjacent_array, edge_list = erdos_renyi_model(N, L)
        node_edge_number = numb_of_links(edge_list)
        distribution = degree_dist(node_edge_number, N)
        avg_degree = average_degree(node_edge_number)
        avg_avg_degree += avg_degree
        c_avg, c_i = clusterCoeff(adjacent_array, node_edge_number=node_edge_number)
        avg_c_avg += c_avg
        avg_c_i += c_i
        keys = []

        for k, v in distribution.items():
            for i in avg_distribution.keys():
                keys.append(i)
            if k in keys:
                avg_distribution[k] += v
            else:
                avg_distribution[k] = types.float64(v)
        distances_from_node = floyd_warshall(edge_list, N)
        dist_of_paths, diameter, avg_path = shortest_path_diameter_avg_path(distances_from_node)
        avg_avg_path += avg_path
        keys = []

        for k, v in dist_of_paths.items():
            for i in avg_dist_of_paths.keys():
                keys.append(i)
            if k in keys:
                avg_dist_of_paths[k] += v
            else:
                avg_dist_of_paths[k] = types.float64(v)
        avg_diameter += diameter
    dist_list.append(avg_distribution)
    avg_list.append(avg_avg_degree / T)
    avg_c_avg_list.append(avg_c_avg / N)
    avg_c_i_list.append(avg_c_i)
    avg_dist_of_paths_list.append(avg_dist_of_paths)
    avg_diameter_list.append(avg_diameter / T)
    avg_path_list.append(avg_avg_path / T)
    return dist_list, avg_list, avg_c_avg_list, avg_c_i_list, avg_dist_of_paths_list, avg_diameter_list, avg_path_list


def all_plots(N, K, beta, p, L, T):
    N = N
    K = K
    beta = beta
    T = T
    L = L
    p = p
    dist_list, avg_list, avg_c_avg_list, avg_c_i_list, avg_dist_of_paths_list, avg_diameter_list, avg_path_list = check_all_models(
        N, K, beta, p, T, L)
    labels = []
    graph = []
    plt.grid(linestyle='dotted', linewidth=1)
    labels.append(
        "Watts-strogatz N= " + str(N) + " k= " + str(K) + " beta= " + str(beta) + " avg= " + str(round(avg_list[0], 3)))
    g = plt.bar(dist_list[0].keys(), np.array(list(dist_list[0].values())) / T, alpha=0.6)
    graph.append(g)

    labels.append("Erdos-Renei-Gilbert N= " + str(N) + " p= " + str(p) + " avg= " + str(round(avg_list[0], 3)))
    g = plt.bar(dist_list[1].keys(), np.array(list(dist_list[1].values())) / T, alpha=0.6)
    graph.append(g)

    labels.append("Erdos-Renei N= " + str(N) + " L= " + str(p) + " avg= " + str(round(avg_list[0], 3)))
    g = plt.bar(dist_list[2].keys(), np.array(list(dist_list[2].values())) / T, alpha=0.6)
    graph.append(g)

    plt.legend(graph, labels=labels, loc="upper right")
    #plt.xlim([0, 100])
    #plt.xticks(range(0, 100, 5))
    plt.xlabel("Degree of node k")
    plt.ylabel("P(k)")
    plt.show()

    graph1 = []
    bins = np.linspace(0, 1, 21)
    labels = []
    for b in range(len(bins) - 1):
        labels.append(str(round(bins[b], 3)) + " - " + str(round(bins[b + 1], 3)))
    mv = [0, 0.01, 0.02]
    labelsl = []
    avg_c_i_list
    ######################
    c_i = list(np.array(avg_c_i_list[0]) / T)
    labelsl.append("Watts-strogatz N= " + str(N) + " k= " + str(K) + " beta= " + str(beta))
    whereBelong = np.digitize(x=c_i, bins=bins[1:], right=True)
    counts = np.zeros(len(bins[1:]))
    for i in whereBelong:
        counts[i] += 1
    cent_bins = []
    for i in range(len(bins) - 1):
        cent_bins.append(np.mean([bins[i], bins[i + 1]]))
    g = plt.bar(np.array(cent_bins) + mv[0], counts / (N), alpha=0.6, width=0.01)
    graph1.append(g)

    c_i = list(np.array(avg_c_i_list[1]) / T)
    labelsl.append("Erdos-Renei-Gilbert N= " + str(N) + " p= " + str(p))
    whereBelong = np.digitize(x=c_i, bins=bins[1:], right=True)
    counts = np.zeros(len(bins[1:]))
    for i in whereBelong:
        counts[i] += 1
    cent_bins = []
    for i in range(len(bins) - 1):
        cent_bins.append(np.mean([bins[i], bins[i + 1]]))
    g = plt.bar(np.array(cent_bins) + mv[1], counts / (N), alpha=0.6, width=0.01)
    graph1.append(g)

    c_i = list(np.array(avg_c_i_list[2]) / T)
    labelsl.append("Erdos-Renei N= " + str(N) + " L= " + str(L))
    whereBelong = np.digitize(x=c_i, bins=bins[1:], right=True)
    counts = np.zeros(len(bins[1:]))
    for i in whereBelong:
        counts[i] += 1
    cent_bins = []
    for i in range(len(bins) - 1):
        cent_bins.append(np.mean([bins[i], bins[i + 1]]))
    g = plt.bar(np.array(cent_bins) + mv[2], counts / (N), alpha=0.6, width=0.01)
    graph1.append(g)
    #
    plt.legend(graph1, labels=labelsl, loc="upper right")
    plt.ylabel("Number of Occurrences")
    plt.xticks(bins[:-1] + 0.035, labels)
    plt.xlabel("Cluster Coefficient")
    plt.grid(linestyle='dotted', linewidth=1)
    plt.xlim([0.0, 1.])
    plt.show()

    labels = []
    plt.grid(linestyle='dotted', linewidth=1)
    mv = [-0.1, 0.0, 0.1]
    labels.append("Watts-strogatz N= " + str(N) + " k= " + str(K) + " beta= " + str(beta))
    g = plt.bar(np.array(list(avg_dist_of_paths_list[0].keys())) + mv[0],
                np.array(list(avg_dist_of_paths_list[0].values())) / T, alpha=0.6, width=0.1)
    graph.append(g)

    labels.append("Erdos-Renei-Gilbert N= " + str(N) + " p= " + str(p))
    g = plt.bar(np.array(list(avg_dist_of_paths_list[1].keys())) + mv[1],
                np.array(list(avg_dist_of_paths_list[1].values())) / T, alpha=0.6, width=0.1)
    graph.append(g)

    labels.append("Erdos-Renei N= " + str(N) + " L= " + str(L))
    g = plt.bar(np.array(list(avg_dist_of_paths_list[2].keys())) + mv[2],
                np.array(list(avg_dist_of_paths_list[2].values())) / T, alpha=0.6, width=0.1)
    graph.append(g)

    plt.legend(graph, labels=labels, loc="upper right")
    plt.xlim([-1, 11])
    plt.xticks(range(-1, 11))
    plt.xlabel("Length of path")
    plt.ylabel("Distribution of path length")
    plt.show()


if __name__ == '__main__':

    brain = nx.read_edgelist("bn/brain.edges")
    G = nx.Graph(brain)
    nx.draw(G)
    plt.title("Brains proteins")
    plt.show()

    mygraph = nx.read_edgelist("facebook_combined.txt")
    G = nx.Graph(mygraph)
    nx.draw(G)
    plt.title("Facebook")
    plt.show()

    data = rfile("facebook_combined.txt")
    node_edge_number = numb_of_links(data)
    distribution = degree_dist(node_edge_number, 4039)
    avg_degree = average_degree(node_edge_number)

    graph = []
    labels = []
    plt.grid(linestyle='dotted', linewidth=1)
    labels.append("avg=" + str(round(avg_degree, 3)))
    g = plt.scatter(distribution.keys(), np.array(list(distribution.values())))
    graph.append(g)
    plt.legend(graph, labels=labels, loc="upper right")
    plt.xlabel("Degree of node k")
    plt.ylabel("P(k)")
    plt.show()

    adjacent_array = np.zeros((4039, 4039))

    for (node1, node2) in data:
        adjacent_array[node1][node2] = 1
        adjacent_array[node2][node1] = 1
    c_avg, c_i = clusterCoeff(adjacent_array, node_edge_number=node_edge_number)

    graph1 = []
    bins = np.linspace(0, 1, 21)
    labels = []
    for b in range(len(bins) - 1):
        labels.append(str(round(bins[b], 3)) + " - " + str(round(bins[b + 1], 3)))
    labelsl = []

    c_i = list(np.array(c_i))
    labelsl.append("avg=" + str(round(c_avg, 3)))
    whereBelong = np.digitize(x=c_i, bins=bins[1:], right=True)
    counts = np.zeros(len(bins[1:]))
    for i in whereBelong:
        counts[i] += 1
    cent_bins = []

    for i in range(len(bins) - 1):
        cent_bins.append(np.mean([bins[i], bins[i + 1]]))
    g = plt.bar(np.array(cent_bins) + 0.01, counts / 4039, alpha=0.6, width=0.04)
    graph1.append(g)
    plt.legend(graph1, labels=labelsl, loc="upper right")
    plt.ylabel("Number of Occurrences", fontsize=16)
    plt.xticks(bins[:-1] + 0.035, labels)
    plt.xlabel("Cluster Coefficient", fontsize=16)
    plt.grid(linestyle='dotted', linewidth=1)
    plt.xlim([0.0, 1.])
    plt.show()

    distances_from_node = floyd_warshall(data, 4039)
    dist_of_paths, diameter, avg_path = shortest_path_diameter_avg_path(distances_from_node)

    labels = []
    plt.grid(linestyle='dotted', linewidth=1)
    labels.append("avg_path=" + str(round(avg_path, 3)) + " diameter= " + str(diameter))
    g = plt.bar(np.array(list(dist_of_paths.keys())), np.array(list(dist_of_paths.values())), width=0.8)
    graph.append(g)
    plt.legend(graph, labels=labels, loc="upper right")
    plt.xlim([-1, 11])
    plt.xticks(range(-1, 11))
    plt.xlabel("Length of path", fontsize=16)
    plt.ylabel("Distribution of path length", fontsize=16)
    plt.show()

    erdos_renyi_graphs_data_plot(numba.typed.List([100, 100, 100, 100, 100]),
                                 numba.typed.List([100, 200, 500, 2000, 3000]), T=1000)
    watts_strogatz_graphs_plots([100, 100, 100, 100, 100], [4, 4, 4, 4, 4], [0.0, 0.25, 0.5, 0.75, 1.0], T=1000)
    watts_strogatz_graphs_plots([100, 100, 100, 100, 100], [4, 10, 25, 35, 80], [0.5, 0.5, 0.5, 0.5, .5], T=1000)
    erdos_renyi_gilbert_graphs_data_plot([100, 100, 100, 100, 100], [0.1, 0.25, 0.5, 0.75, 0.9], 1000)
    all_plots(100, 40, 0.5, 0.6, 3000, 1000)
