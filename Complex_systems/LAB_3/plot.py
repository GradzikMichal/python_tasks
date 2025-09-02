import matplotlib.pyplot as plt
import matplotlib.colors as colors
from random import randint
import numpy as np
from matplotlib.ticker import MultipleLocator
import scipy.stats as stats
from scipy.optimize import curve_fit


def draw_perc_three_color(perc_lattice, L):
    perc_lattice = perc_lattice
    for x in range(L):
        for y in range(L):
            if perc_lattice[x][y] > 1:
                perc_lattice[x][y] = 2

    fig, ax = plt.subplots()
    cmp = colors.ListedColormap(['white', 'grey', 'red'])
    ax.pcolormesh(range(0, L), range(0, L), perc_lattice, cmap=cmp, vmin=0, vmax=2)
    fig.set_size_inches(9.5, 9.5)
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.title("Three color 2-D Heat Map")
    plt.show()


def draw_perc_two_color(perc_lattice, L):
    perc_lattice = perc_lattice
    fig, ax = plt.subplots()
    cmp = colors.ListedColormap(['white', 'black'])
    ax.pcolormesh(range(0, L), range(0, L), perc_lattice, cmap=cmp, vmin=0, vmax=1)
    fig.set_size_inches(9.5, 9.5)
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.title("Two color 2-D Heat Map")
    plt.show()


def draw_perc_two_color_with_text(perc_lattice, L, p):
    perc_lattice = perc_lattice
    fig, ax = plt.subplots()
    cmp = colors.ListedColormap(['white', 'black'])
    ax.pcolormesh(range(0, L), range(0, L), perc_lattice, cmap=cmp, vmin=0, vmax=1)
    fig.set_size_inches(9.5, 9.5)
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.title("Two color percolation for " + str(round(p, 2)))
    if L > 50:
        for i in range(perc_lattice.shape[0]):
            for j in range(perc_lattice.shape[1]):
                if perc_lattice[i][j] == 0:
                    plt.text(j, i, perc_lattice[i][j], ha='center', va='center')
                else:
                    plt.text(j, i, perc_lattice[i][j], color="white", ha='center', va='center')
    plt.savefig('plots/two_color_perc_' + str(round(p, 2)) + '.png', bbox_inches='tight')
    plt.show()


def single_p_critical(x, y):
    plt.plot(x, y)
    plt.title("Wrapping probability as a function of the occupation probability for single lattice size")
    plt.xlabel('Occupation probability p')
    plt.ylabel('Wrapping Probability W(p)')
    plt.show()


def draw_clusters_colors(clusters, array_of_clusters, L, p):
    fig, ax = plt.subplots()
    color = set()
    while len(color) < (len(array_of_clusters) - 1):
        clr = '#%06X' % randint(0, 0xFFFFFF)
        if clr != "#000000":
            color.update({clr})
    color = ["#000000"] + list(color)
    cmp = colors.ListedColormap(list(color))
    bounds = array_of_clusters + [array_of_clusters[-1] + 1]
    norm = colors.BoundaryNorm(bounds, cmp.N)
    ticks = list(np.array(array_of_clusters) + 0.5)
    clr = ax.pcolormesh(range(0, L), range(0, L), clusters, cmap=cmp, norm=norm)
    cb = plt.colorbar(clr, cmap=cmp, norm=norm, ticks=ticks)
    cb.set_ticklabels(array_of_clusters)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # for i in range(clusters.shape[0]):
    #    for j in range(clusters.shape[1]):
    #        plt.text(j, i, clusters[i][j], color="white", ha='center', va='center')

    plt.title("Clusters in the system for p: " + str(round(p, 2)))
    plt.savefig('plots/color_clusters_' + str(round(p, 2)) + '.png', bbox_inches='tight')
    plt.show()


def draw_perc(perc_lattice):
    plt.imshow(perc_lattice)
    plt.title("2-D Heat Map")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def cluster_size_distribution():
    plots = [
        "output_data/Dist_p0.10L100T10000.txt",
        "output_data/Dist_p0.20L100T10000.txt",
        "output_data/Dist_p0.30L100T10000.txt",
        "output_data/Dist_p0.40L100T10000.txt",
        "output_data/Dist_p0.50L100T10000.txt",
        "output_data/Dist_p0.59L100T10000.txt",
        "output_data/Dist_p0.60L100T10000.txt",
        "output_data/Dist_p0.70L100T10000.txt",
        "output_data/Dist_p0.80L100T10000.txt",
        "output_data/Dist_p0.90L100T10000.txt"
    ]
    data = dict()
    for file in plots:
        with open(file, 'r') as f:
            data[float(file[18:22])] = dict()
            for line in f:
                l = line.split(sep="  ")
                data[float(file[18:22])][int(l[0])] = float(l[1])
    fig, ax = plt.subplots(1, 3)
    ax[0].grid()
    ax1 = ax[0].scatter(data[0.10].keys(), np.log(np.array(list(data[0.10].values())) / 10), c='pink', marker="s", s=30)
    ax2 = ax[0].scatter(data[0.20].keys(), np.log(np.array(list(data[0.20].values())) / 10), c='red', marker="*", s=30)
    ax3 = ax[0].scatter(data[0.30].keys(), np.log(np.array(list(data[0.30].values())) / 10), c='green', marker=".",
                        s=30)
    ax4 = ax[0].scatter(data[0.40].keys(), np.log(np.array(list(data[0.40].values())) / 10), c='blue', marker="v", s=30)
    ax5 = ax[0].scatter(data[0.50].keys(), np.log(np.array(list(data[0.50].values())) / 10), c='purple', marker="X",
                        s=30)
    ax[0].set_xlim([0, 50])
    ax[0].set_ylim([-16, -2])
    ax[0].set_yticks(list(range(-16, -2)))

    ax[0].xaxis.set_minor_locator(MultipleLocator(1))
    ax[0].set_xlabel("s")
    ax[0].set_title('$n _{p<p _{c}}$', fontsize=12)
    ax[0].set_ylabel('ln $n _{s}$', fontsize=12)

    d = np.array(list(data[0.59].values()))
    x = list(data[0.59].keys())
    ax[1].grid()
    ax6 = ax[1].scatter(x, d * 100 * 100, c='black', marker="p", s=30)

    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel("s")
    ax[1].set_title('$n _{p=p _{c}}$', fontsize=12)
    ax[1].set_ylabel('$ n _{s} L ^{2}$', fontsize=12)
    ax[2].grid()
    ax7 = ax[2].scatter(data[0.60].keys(), np.log(np.array(list(data[0.60].values())) / 100), c='gray', marker="P",
                        s=30)
    ax8 = ax[2].scatter(data[0.70].keys(), np.log(np.array(list(data[0.70].values())) / 100), c='darkblue', marker="h",
                        s=30)
    ax9 = ax[2].scatter(data[0.80].keys(), np.log(np.array(list(data[0.80].values())) / 100), c='yellow', marker="H",
                        s=30)
    ax10 = ax[2].scatter(data[0.90].keys(), np.log(np.array(list(data[0.90].values())) / 100), c='orange', marker="x",
                         s=30)
    ax[2].set_xlim([0, 200])
    ax[2].set_ylim([-16, -4])
    ax[2].set_yticks(list(range(-16, -4)))
    ax[2].set_xticks(list(range(0, 201, 50)))
    ax[2].xaxis.set_minor_locator(MultipleLocator(10))
    ax[2].set_xlabel("s")
    ax[2].set_title('$n _{p>p _{c}}$', fontsize=12)
    ax[2].set_ylabel('ln $n _{s}$', fontsize=12)
    labels = ['p = 0.1', 'p = 0.2', 'p = 0.3', 'p = 0.4', 'p = 0.5', 'p = 0.5926', 'p = 0.6', 'p = 0.7', 'p = 0.8',
              'p = 0.9']
    fig.legend([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10], labels=labels, loc="upper right")
    fig.suptitle("Cluster size distribution")
    fig.set_size_inches(19, 10)
    plt.savefig('plots/graph_of_cluster_size_dist' + '.png', bbox_inches='tight', dpi=100)
    plt.show()


def cluster_size_distribution_full_plots():
    plots = [
        "output_data/Dist_p0.10L100T10000.txt",
        "output_data/Dist_p0.20L100T10000.txt",
        "output_data/Dist_p0.30L100T10000.txt",
        "output_data/Dist_p0.40L100T10000.txt",
        "output_data/Dist_p0.50L100T10000.txt",
        "output_data/Dist_p0.59L100T10000.txt",
        "output_data/Dist_p0.60L100T10000.txt",
        "output_data/Dist_p0.70L100T10000.txt",
        "output_data/Dist_p0.80L100T10000.txt",
        "output_data/Dist_p0.90L100T10000.txt"
    ]
    data = dict()
    for file in plots:
        with open(file, 'r') as f:
            data[float(file[18:22])] = dict()
            for line in f:
                l = line.split(sep="  ")
                data[float(file[18:22])][int(l[0])] = float(l[1])
    fig, ax = plt.subplots(1, 3)
    ax[0].grid()
    ax1 = ax[0].scatter(data[0.10].keys(), np.log(np.array(list(data[0.10].values())) / 10), c='pink', marker="s", s=30)
    ax2 = ax[0].scatter(data[0.20].keys(), np.log(np.array(list(data[0.20].values())) / 10), c='red', marker="*", s=30)
    ax3 = ax[0].scatter(data[0.30].keys(), np.log(np.array(list(data[0.30].values())) / 10), c='green', marker=".",
                        s=30)
    ax4 = ax[0].scatter(data[0.40].keys(), np.log(np.array(list(data[0.40].values())) / 10), c='blue', marker="v", s=30)
    ax5 = ax[0].scatter(data[0.50].keys(), np.log(np.array(list(data[0.50].values())) / 10), c='purple', marker="X",
                        s=30)
    # ax[0].set_xlim([0, 50])
    # ax[0].set_ylim([-16, 0])
    ax[0].set_yticks(list(range(-19, -2)))

    ax[0].xaxis.set_minor_locator(MultipleLocator(1))
    ax[0].set_xlabel("s")
    ax[0].set_title('$n _{p<p _{c}}$', fontsize=12)
    ax[0].set_ylabel('ln $n _{s}$', fontsize=12)

    d = np.array(list(data[0.59].values()))
    x = list(data[0.59].keys())
    ax[1].grid()
    ax6 = ax[1].scatter(x, d * 100 * 100, c='black', marker="p", s=30)

    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel("s")
    ax[1].set_title('$n _{p=p _{c}}$', fontsize=12)
    ax[1].set_ylabel('$ n _{s} L ^{2}$', fontsize=12)
    ax[2].grid()
    ax7 = ax[2].scatter(data[0.60].keys(), np.log(np.array(list(data[0.60].values())) / 100), c='gray', marker="P",
                        s=30)
    ax8 = ax[2].scatter(data[0.70].keys(), np.log(np.array(list(data[0.70].values())) / 100), c='darkblue', marker="h",
                        s=30)
    ax9 = ax[2].scatter(data[0.80].keys(), np.log(np.array(list(data[0.80].values())) / 100), c='yellow', marker="H",
                        s=30)
    ax10 = ax[2].scatter(data[0.90].keys(), np.log(np.array(list(data[0.90].values())) / 100), c='orange', marker="x",
                         s=30)
    ax[2].set_yticks(list(range(-20, -4)))
    ax[2].set_xticks(list(range(0, 10000, 1000)))
    ax[2].xaxis.set_minor_locator(MultipleLocator(10))
    ax[2].set_xlabel("s")
    ax[2].set_title('$n _{p>p _{c}}$', fontsize=12)
    ax[2].set_ylabel('ln $n _{s}$', fontsize=12)
    labels = ['p = 0.1', 'p = 0.2', 'p = 0.3', 'p = 0.4', 'p = 0.5', 'p = 0.5926', 'p = 0.6', 'p = 0.7', 'p = 0.8',
              'p = 0.9']
    fig.legend([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10], labels=labels, loc="upper right")
    fig.suptitle("Cluster size distribution")
    fig.set_size_inches(19, 10)
    plt.savefig('plots/graph_of_cluster_size_dist_full.png', bbox_inches='tight', dpi=100)
    plt.show()


def analisys_of_cluster_Sizes_for_high_p_with_fit():
    plots = [
        "output_data/Dist_p0.70L100T10000.txt",
        "output_data/Dist_p0.80L100T10000.txt",
        "output_data/Dist_p0.90L100T10000.txt"
    ]
    data = dict()
    for file in plots:
        with open(file, 'r') as f:
            data[float(file[18:22])] = dict()
            for line in f:
                l = line.split(sep="  ")
                data[float(file[18:22])][int(l[0])] = float(l[1])
    data_with_list = dict()
    for prob_key in data.keys():
        list_of_tuples = []
        for (size, val) in zip(data[prob_key].keys(), data[prob_key].values()):
            list_of_tuples.append((size, val))
        sorted_list = sorted(list_of_tuples, key=lambda tup: tup[0])
        data_with_list[prob_key] = sorted_list

    fig, ax = plt.subplots(1, 3)
    ax[0].grid()

    x_data = []
    y_data = []
    for (x, y) in data_with_list[0.70]:
        x_data.append(x)
        y_data.append(y)
    x_data = x_data[180:]
    y_data = y_data[180:]
    ax1 = ax[0].scatter(x_data, y_data, c='pink', marker="s",
                        s=30, label='p=0.7')
    result1 = curve_fit(norm_dist, x_data, y_data)
    print(result1)
    mi = result1[0][1]
    sigma = result1[0][0]
    fitted_y = []
    for x in range(6250, 7750):
        fitted_y.append(np.log(norm_dist(x, sigma, mi) / 100))
    print(fitted_y)
    ax[0].plot(range(6250, 7750), fitted_y, '--',
               label="p=0.7 sigma=" + str(round(sigma, 3)) + " mi=" + str(round(mi, 3)),
               c='pink')

    fig, ax = plt.subplots(1, 3)
    ax[0].grid()
    ax[0].xaxis.set_minor_locator(MultipleLocator(100))
    ax[0].set_xlabel("s")
    ax[0].set_yticks(list(range(-19, -13)))
    ax[0].set_xlim([6250, 7750])
    ax[0].set_title('$n _{p=0.70}$', fontsize=12)
    ax[0].set_ylabel('ln $n _{s}$', fontsize=12)
    ax[0].set_ylim([-20, -13])

    ax[1].grid()
    ax2 = ax[1].scatter(data[0.80].keys(), np.log(np.array(list(data[0.80].values())) / 100), c='yellow', marker="p",
                        s=30)
    ax[1].set_xlabel("s")
    ax[1].xaxis.set_minor_locator(MultipleLocator(50))
    ax[1].set_xlim([7700, 8300])
    ax[1].set_yticks(list(range(-18, -10)))
    ax[1].set_ylim([-19, -10])
    ax[1].set_title('$n _{p=0.80}$', fontsize=12)
    ax[1].set_ylabel('ln $n _{s}$', fontsize=12)

    ax[2].grid()
    ax3 = ax[2].scatter(data[0.90].keys(), np.log(np.array(list(data[0.90].values())) / 100), c='orange', marker="P",
                        s=30)
    ax[2].set_xlabel("s")
    ax[2].xaxis.set_minor_locator(MultipleLocator(25))
    ax[2].set_title('$n _{p=0.90}$', fontsize=12)
    ax[2].set_ylabel('ln $n _{s}$', fontsize=12)
    ax[2].set_xlim([8800, 9200])
    ax[2].set_yticks(list(range(-16, -8)))
    ax[2].set_ylim([-17, -8])

    labels = ['p = 0.7', 'p = 0.8', 'p = 0.9']
    fig.legend([ax1, ax2, ax3], labels=labels, loc="upper right")
    fig.suptitle("Analysis of cluster size distribution")
    fig.set_size_inches(19, 10)
    plt.savefig('plots/analysis_of_cluster_size_dist.png', bbox_inches='tight', dpi=100)
    plt.show()


def norm_dist(x, sig, u):
    return 1/np.sqrt(2*np.pi*sig**2)*np.exp(-(x-u)**2/2*sig**2)


def analisys_of_cluster_Sizes_for_high_p():
    plots = [
        "output_data/Dist_p0.70L100T10000.txt",
        "output_data/Dist_p0.80L100T10000.txt",
        "output_data/Dist_p0.90L100T10000.txt"
    ]
    data = dict()
    for file in plots:
        with open(file, 'r') as f:
            data[float(file[18:22])] = dict()
            for line in f:
                l = line.split(sep="  ")
                data[float(file[18:22])][int(l[0])] = float(l[1])
    fig, ax = plt.subplots(1, 3)
    ax[0].grid()
    ax1 = ax[0].scatter(data[0.70].keys(), np.log(np.array(list(data[0.70].values())) / 100), c='darkblue', marker="s",
                        s=30)
    ax[0].xaxis.set_minor_locator(MultipleLocator(100))
    ax[0].set_xlabel("s")
    ax[0].set_yticks(list(range(-19, -13)))
    ax[0].set_xlim([6250, 7750])
    ax[0].set_title('$n _{p=0.70}$', fontsize=12)
    ax[0].set_ylabel('ln $n _{s}$', fontsize=12)
    ax[0].set_ylim([-20, -13])

    ax[1].grid()
    ax2 = ax[1].scatter(data[0.80].keys(), np.log(np.array(list(data[0.80].values())) / 100), c='yellow', marker="p",
                        s=30)
    ax[1].set_xlabel("s")
    ax[1].xaxis.set_minor_locator(MultipleLocator(50))
    ax[1].set_xlim([7700, 8300])
    ax[1].set_yticks(list(range(-18, -10)))
    ax[1].set_ylim([-19, -10])
    ax[1].set_title('$n _{p=0.80}$', fontsize=12)
    ax[1].set_ylabel('ln $n _{s}$', fontsize=12)

    ax[2].grid()
    ax3 = ax[2].scatter(data[0.90].keys(), np.log(np.array(list(data[0.90].values())) / 100), c='orange', marker="P",
                        s=30)
    ax[2].set_xlabel("s")
    ax[2].xaxis.set_minor_locator(MultipleLocator(25))
    ax[2].set_title('$n _{p=0.90}$', fontsize=12)
    ax[2].set_ylabel('ln $n _{s}$', fontsize=12)
    ax[2].set_xlim([8800, 9200])
    ax[2].set_yticks(list(range(-16, -8)))
    ax[2].set_ylim([-17, -8])

    labels = ['p = 0.7', 'p = 0.8', 'p = 0.9']
    fig.legend([ax1, ax2, ax3], labels=labels, loc="upper right")
    fig.suptitle("Analysis of cluster size distribution")
    fig.set_size_inches(19, 10)
    plt.savefig('plots/analysis_of_cluster_size_dist.png', bbox_inches='tight',  dpi=100)
    plt.show()


def plot_pcritical():
    plots = [
        "output_data/Ave_L10T10000.txt",
        "output_data/Ave_L50T10000.txt",
        "output_data/Ave_L100T10000.txt"
    ]
    sizes = [10, 50, 100]
    data = dict()
    for file, size in zip(plots, sizes):
        with open(file, 'r') as f:
            data[size] = dict()
            for line in f:
                l = line.split(sep="  ")
                data[size][float(l[0])] = float(l[1])
    l1 = plt.plot(list(data[10].keys())[:-1], list(data[10].values())[:-1], '.-')
    l2 = plt.plot(list(data[50].keys())[:-1], list(data[50].values())[:-1], '^-')
    l3 = plt.plot(list(data[100].keys())[:-1], list(data[100].values())[:-1], 's-')
    labels = ["L = 10", "L = 50", "L = 100"]
    plt.legend([l1, l2, l3], labels=labels, loc="upper right")
    plt.xlabel("Probability p")
    plt.ylabel("Probability that path exists")
    plt.title("Probability that path connecting the first and the last row exist.")
    plt.minorticks_on()
    plt.yticks(list(np.linspace(0, 1, 11)))
    plt.xticks(list(np.linspace(0, 1, 11)))
    plt.grid(which='major')
    plt.savefig('plots/graph_of_pcritical' + '.png', bbox_inches='tight')
    plt.show()


def avg_max_size_cluster():
    plots = [
        "output_data/Ave_L10T10000.txt",
        "output_data/Ave_L50T10000.txt",
        "output_data/Ave_L100T10000.txt"
    ]
    sizes = [10, 50, 100]
    data = dict()
    for file, size in zip(plots, sizes):
        with open(file, 'r') as f:
            data[size] = dict()
            for line in f:
                l = line.split(sep="  ")
                data[size][float(l[0])] = float(l[2])
    plt.figure(figsize=(9, 9))
    l1 = plt.plot(list(data[10].keys())[:-1], list(data[10].values())[:-1], '.-')
    l2 = plt.plot(list(data[50].keys())[:-1], list(data[50].values())[:-1], '^-')
    l3 = plt.plot(list(data[100].keys())[:-1], list(data[100].values())[:-1], 's-')
    labels = ["L = 10", "L = 50", "L = 100"]
    plt.legend([l1, l2, l3], labels=labels, loc="upper right")
    plt.xlabel("Probability p")
    plt.ylabel('<$s _{max}$>')
    plt.title("Average size of the maximum cluster")
    plt.minorticks_on()
    plt.xticks(list(np.linspace(0, 1, 11)))
    plt.grid(which='major')
    plt.savefig('plots/graph_of_avg_size' + '.png', bbox_inches='tight')
    plt.show()


def normalize_avg_max_size_cluster():
    plots = [
        "output_data/Ave_L10T10000.txt",
        "output_data/Ave_L50T10000.txt",
        "output_data/Ave_L100T10000.txt"
    ]
    sizes = [10, 50, 100]
    data = dict()
    for file, size in zip(plots, sizes):
        with open(file, 'r') as f:
            data[size] = dict()
            for line in f:
                l = line.split(sep="  ")
                data[size][float(l[0])] = float(l[2])
    max_10 = max(list(data[10].values())[:-1])
    max_50 = max(list(data[50].values())[:-1])
    max_100 = max(list(data[100].values())[:-1])
    plt.figure(figsize=(9, 9))
    l1 = plt.plot(list(data[10].keys())[:-1], np.array(list(data[10].values()))[:-1] / max_10, '.-')
    l2 = plt.plot(list(data[50].keys())[:-1], np.array(list(data[50].values()))[:-1] / max_50, '^-')
    l3 = plt.plot(list(data[100].keys())[:-1], np.array(list(data[100].values()))[:-1] / max_100, 's-')
    labels = ["L = 10", "L = 50", "L = 100"]
    plt.legend([l1, l2, l3], labels=labels, loc="upper right")
    plt.xlabel("Probability p")
    plt.ylabel('Normalized <$s _{max}$>')
    plt.title("Normalized average size of the maximum cluster")
    plt.minorticks_on()
    plt.xticks(list(np.linspace(0, 1, 11)))
    plt.yticks(list(np.linspace(0, 1, 11)))

    plt.grid(which='major')

    plt.savefig('plots/graph_of_normalized_avg_size' + '.png', bbox_inches='tight')
    plt.show()


def power_law_p_small(s, thetha, a):
    return (s ** -thetha) * np.e ** (a * s)


def power_law_p_crit(s, gamma):
    return s ** -gamma


def power_law_p_big(s, b, d):
    return np.e ** (-b * s ** (1 - 1 / d))


def fitting_to_plot():
    plots = [
        "output_data/Dist_p0.10L100T10000.txt",
        "output_data/Dist_p0.20L100T10000.txt",
        "output_data/Dist_p0.30L100T10000.txt",
        "output_data/Dist_p0.40L100T10000.txt",
        "output_data/Dist_p0.50L100T10000.txt",
        "output_data/Dist_p0.59L100T10000.txt",
        "output_data/Dist_p0.60L100T10000.txt",
        "output_data/Dist_p0.70L100T10000.txt",
        "output_data/Dist_p0.80L100T10000.txt",
        "output_data/Dist_p0.90L100T10000.txt"
    ]
    data = dict()
    for file in plots:
        with open(file, 'r') as f:
            data[float(file[18:22])] = dict()
            for line in f:
                l = line.split(sep="  ")
                data[float(file[18:22])][int(l[0])] = float(l[1])
    data_with_list = dict()
    for prob_key in data.keys():
        list_of_tuples = []
        for (size, val) in zip(data[prob_key].keys(), data[prob_key].values()):
            list_of_tuples.append((size, val))
        sorted_list = sorted(list_of_tuples, key=lambda tup: tup[0])
        data_with_list[prob_key] = sorted_list

    fig, ax = plt.subplots(1, 3)
    ax[0].grid()
    ax1 = ax[0].scatter(data[0.10].keys(), np.log(np.array(list(data[0.10].values())) / 10), c='pink', marker="s",
                        s=30, label='p=0.1')
    x_data = []
    y_data = []
    for (x, y) in data_with_list[0.10]:
        x_data.append(x)
        y_data.append(y)
    x_data = x_data[3:14]
    y_data = y_data[3:14]
    result1 = curve_fit(power_law_p_small, x_data, y_data)
    a = result1[0][1]
    theta = result1[0][0]
    fitted_y = []
    for x in range(1, 50):
        fitted_y.append(np.log(power_law_p_small(x, theta, a) / 10))
    ax[0].plot(range(1, 50), fitted_y, '--', label="p=0.1 a=" + str(round(a, 3)) + " theta=" + str(round(theta, 3)),
               c='pink')
    ax2 = ax[0].scatter(data[0.20].keys(), np.log(np.array(list(data[0.20].values())) / 10), c='red', marker="*",
                        s=30, label='p=0.2')
    x_data = []
    y_data = []
    for (x, y) in data_with_list[0.20]:
        x_data.append(x)
        y_data.append(y)
    x_data = x_data[3:]
    y_data = y_data[3:]
    result1 = curve_fit(power_law_p_small, x_data, y_data)
    a = result1[0][1]
    theta = result1[0][0]
    fitted_y = []
    for x in range(1, 50):
        fitted_y.append(np.log(power_law_p_small(x, theta, a) / 10))
    ax[0].plot(range(1, 50), fitted_y, '-.', label="p=0.2 a=" + str(round(a, 3)) + " theta=" + str(round(theta, 3)),
               c='red')

    ax3 = ax[0].scatter(data[0.30].keys(), np.log(np.array(list(data[0.30].values())) / 10), c='green', marker=".",
                        s=30, label='p=0.3')
    x_data = []
    y_data = []
    for (x, y) in data_with_list[0.30]:
        x_data.append(x)
        y_data.append(y)
    x_data = x_data[10:]
    y_data = y_data[10:]
    result1 = curve_fit(power_law_p_small, x_data, y_data)
    a = result1[0][1]
    theta = result1[0][0]
    fitted_y = []
    for x in range(1, 50):
        fitted_y.append(np.log(power_law_p_small(x, theta, a) / 10))
    ax[0].plot(range(1, 50), fitted_y, label="p=0.3 a=" + str(round(a, 3)) + " theta=" + str(round(theta, 3)),
               c='green')
    ax4 = ax[0].scatter(data[0.40].keys(), np.log(np.array(list(data[0.40].values())) / 10), c='blue', marker="v",
                        s=30, label='p=0.4')
    x_data = []
    y_data = []
    for (x, y) in data_with_list[0.40]:
        x_data.append(x)
        y_data.append(y)
    x_data = x_data[12:]
    y_data = y_data[12:]
    result1 = curve_fit(power_law_p_small, x_data, y_data)
    a = result1[0][1]
    theta = result1[0][0]
    fitted_y = []
    for x in range(1, 50):
        fitted_y.append(np.log(power_law_p_small(x, theta, a) / 10))
    ax[0].plot(range(1, 50), fitted_y, ':', label="p=0.4 a=" + str(round(a, 3)) + " theta=" + str(round(theta, 3)),
               c='gold', lw=3)
    ax5 = ax[0].scatter(data[0.50].keys(), np.log(np.array(list(data[0.50].values())) / 10), c='purple', marker="X",
                        s=30, label='p=0.5')
    x_data = []
    y_data = []
    for (x, y) in data_with_list[0.50]:
        x_data.append(x)
        y_data.append(y)
    x_data = x_data[15:50]
    y_data = y_data[15:50]
    result1 = curve_fit(power_law_p_small, x_data, y_data)
    a = result1[0][1]
    theta = result1[0][0]
    fitted_y = []
    for x in range(1, 50):
        fitted_y.append(np.log(power_law_p_small(x, theta, a) / 10))
    ax[0].plot(range(1, 50), fitted_y, '--', label="p=0.5 a=" + str(round(a, 3)) + " theta=" + str(round(theta, 3)),
               c='orange', dashes=(5, 2), lw=3)

    ax[0].set_xlim([0, 50])
    ax[0].set_ylim([-16, -2])
    ax[0].set_yticks(list(range(-16, -2)))

    ax[0].xaxis.set_minor_locator(MultipleLocator(1))
    ax[0].set_xlabel("s")
    ax[0].set_title('$n _{p<p _{c}}$', fontsize=12)
    ax[0].set_ylabel('ln $n _{s}$', fontsize=12)

    d = np.array(list(data[0.59].values()))
    x = list(data[0.59].keys())
    ax[1].grid()
    ax6 = ax[1].scatter(x, d * 100 * 100, c='black', marker="p", s=30, label="p=0.59")
    x_data = []
    y_data = []
    for (x, y) in data_with_list[0.59]:
        x_data.append(x)
        y_data.append(y)
    x_data = x_data[60:90]
    y_data = y_data[60:90]
    result1 = curve_fit(power_law_p_crit, x_data, y_data)
    gamma = result1[0][0]
    fitted_y = []
    for x in range(0, 1000):
        fitted_y.append(power_law_p_crit(x, gamma) * 100 * 100)
    ax[1].plot(range(0, 1000), fitted_y, label="p=0.59 gamma=" + str(round(gamma, 3)), c='blue', lw=3)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel("s")
    ax[1].set_title('$n _{p=p _{c}}$', fontsize=12)
    ax[1].set_ylabel('$ n _{s} L ^{2}$', fontsize=12)
    ax[2].grid()
    ax7 = ax[2].scatter(data[0.60].keys(), np.log(np.array(list(data[0.60].values())) / 100), c='gray', marker="P",
                        s=30, label="p=0.6")
    x_data = []
    y_data = []
    for (x, y) in data_with_list[0.60]:
        x_data.append(x)
        y_data.append(y)
    x_data = x_data[30:200]
    y_data = y_data[30:200]
    result1 = curve_fit(power_law_p_big, x_data, y_data)
    d = result1[0][1]
    b = result1[0][0]
    fitted_y = []
    for x in range(1, 200):
        fitted_y.append(np.log(power_law_p_big(x, b, d) / 100))
    ax[2].plot(range(1, 200), fitted_y, '--', label="p=0.6 b=" + str(round(b, 3)) + " d=" + str(round(d, 3)), c='red',
               dashes=(5, 2))
    ax8 = ax[2].scatter(data[0.70].keys(), np.log(np.array(list(data[0.70].values())) / 100), c='darkblue',
                        marker="h",
                        s=30, label="p=0.7")
    x_data = []
    y_data = []
    for (x, y) in data_with_list[0.70]:
        x_data.append(x)
        y_data.append(y)
    x_data = x_data[30:200]
    y_data = y_data[30:200]
    result1 = curve_fit(power_law_p_big, x_data, y_data)
    d = result1[0][1]
    b = result1[0][0]
    fitted_y = []
    for x in range(1, 200):
        fitted_y.append(np.log(power_law_p_big(x, b, d) / 100))
    ax[2].plot(range(1, 200), fitted_y, '--', label="p=0.7 b=" + str(round(b, 3)) + " d=" + str(round(d, 3)), c='lime',
               lw=3)
    ax9 = ax[2].scatter(data[0.80].keys(), np.log(np.array(list(data[0.80].values())) / 100), c='yellow',
                        marker="H",
                        s=30, label="p=0.8")
    x_data = []
    y_data = []
    for (x, y) in data_with_list[0.80]:
        x_data.append(x)
        y_data.append(y)
    x_data = x_data[5:30]
    y_data = y_data[5:30]
    result1 = curve_fit(power_law_p_big, x_data, y_data)
    d = result1[0][1]
    b = result1[0][0]
    fitted_y = []
    for x in range(1, 200):
        fitted_y.append(np.log(power_law_p_big(x, b, d) / 100))
    ax[2].plot(range(1, 200), fitted_y, '-.', label="p=0.8 b=" + str(round(b, 3)) + " d=" + str(round(d, 3)), c='blue')
    ax10 = ax[2].scatter(data[0.90].keys(), np.log(np.array(list(data[0.90].values())) / 100), c='orange',
                         marker="x",
                         s=30, label='p=0.9')
    x_data = []
    y_data = []
    for (x, y) in data_with_list[0.90]:
        x_data.append(x)
        y_data.append(y)
    x_data = x_data[0:9]
    y_data = y_data[0:9]
    result1 = curve_fit(power_law_p_big, x_data, y_data)
    d = result1[0][1]
    b = result1[0][0]
    fitted_y = []
    for x in range(1, 200):
        fitted_y.append(np.log(power_law_p_big(x, b, d) / 100))
    ax[2].plot(range(1, 200), fitted_y, ':', label="p=0.9 b=" + str(round(b, 3)) + " d=" + str(round(d, 3)), c='blue')
    ax[2].set_xlim([0, 200])
    ax[2].set_ylim([-16, -4])
    ax[2].set_yticks(list(range(-16, -4)))
    ax[2].set_xticks(list(range(0, 201, 50)))
    ax[2].xaxis.set_minor_locator(MultipleLocator(10))
    ax[2].set_xlabel("s")
    ax[2].set_title('$n _{p>p _{c}}$', fontsize=12)
    ax[2].set_ylabel('ln $n _{s}$', fontsize=12)
    # labels = ['p = 0.1', 'p = 0.2', 'p = 0.3', 'p = 0.4', 'p = 0.5', 'p = 0.5926', 'p = 0.6', 'p = 0.7', 'p = 0.8',
    #          'p = 0.9']
    # fig.legend([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10], labels=labels, loc="upper right")
    fig.legend()
    fig.suptitle("Cluster size distribution")
    fig.set_size_inches(19, 10)
    plt.savefig('plots/graph_of_cluster_size_dist_with_fit' + '.png', bbox_inches='tight',  dpi=100)
    plt.show()


if __name__ == "__main__":
    fitting_to_plot()
    analisys_of_cluster_Sizes_for_high_p_with_fit()
    cluster_size_distribution()
    cluster_size_distribution_full_plots()
    plot_pcritical()
    avg_max_size_cluster()
    normalize_avg_max_size_cluster()
