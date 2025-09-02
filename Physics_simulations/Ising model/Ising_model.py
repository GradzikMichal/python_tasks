# not optimal code, maybe doing it on gpu would be better

from numba import njit
import random
import math
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from matplotlib import colors
import os
from pathlib import Path

RESULTS_DIR = Path(os.path.abspath(__file__)).parent / 'results'


@njit
def calculate_energy_difference(function_lattice, row, column, function_rows, function_columns):
    function_sum = 0
    i = row
    j = column
    if i == 0:
        if j == 0:
            function_sum += function_lattice[i][j] * (
                    function_lattice[i][j + 1] + function_lattice[i][function_columns - 1] +
                    function_lattice[i + 1][j] + function_lattice[function_rows - 1][j])
        elif j < function_columns - 1:
            function_sum += function_lattice[i][j] * (
                        function_lattice[i][j + 1] + function_lattice[i][j - 1] + function_lattice[i + 1][j] +
                        function_lattice[function_rows - 1][j])
        else:
            function_sum += function_lattice[i][j] * (
                        function_lattice[i][0] + function_lattice[i][j - 1] + function_lattice[i + 1][j] +
                        function_lattice[function_rows - 1][j])
    elif i < function_rows - 1:
        if j == 0:
            function_sum += function_lattice[i][j] * (
                        function_lattice[i][j + 1] + function_lattice[i][function_columns - 1] +
                        function_lattice[i + 1][j] + function_lattice[i - 1][j])
        elif j < function_columns - 1:
            function_sum += function_lattice[i][j] * (
                        function_lattice[i][j + 1] + function_lattice[i + 1][j] + function_lattice[i][j - 1] +
                        function_lattice[i - 1][j])
        else:
            function_sum += function_lattice[i][j] * (
                        function_lattice[i][0] + function_lattice[i + 1][j] + function_lattice[i][j - 1] +
                        function_lattice[i - 1][j])
    else:
        if j == 0:
            function_sum += function_lattice[i][j] * (
                        function_lattice[i][j + 1] + function_lattice[0][j] +
                        function_lattice[i][function_columns - 1] + function_lattice[i - 1][j])
        elif j < function_columns - 1:
            function_sum += function_lattice[i][j] * (
                        function_lattice[i][j + 1] + function_lattice[0][j] + function_lattice[i][j - 1] +
                        function_lattice[i - 1][j])
        else:
            function_sum += function_lattice[i][j] * (
                        function_lattice[i][0] + function_lattice[0][j] + function_lattice[i][j - 1] +
                        function_lattice[i - 1][j])
    return function_sum * 2


@njit
def calculate_system_energy(function_lattice, function_rows, function_columns):
    function_sum = 0
    for i in range(0, function_rows):
        for j in range(0, function_columns):
            if i == 0 or i < function_rows - 1:
                if j == 0:
                    function_sum += function_lattice[i][j] * function_lattice[i][j + 1] + function_lattice[i][j] * \
                                    function_lattice[i + 1][j] + function_lattice[i][j] * \
                                    function_lattice[i][function_columns - 1] + function_lattice[i][j] * \
                                    function_lattice[function_rows - 1][j]
                elif j < function_columns - 1:
                    function_sum += function_lattice[i][j] * function_lattice[i][j + 1] + function_lattice[i][j] * \
                                    function_lattice[i + 1][j] + function_lattice[i][j] * \
                                    function_lattice[i][j - 1] + function_lattice[i][j] * \
                                    function_lattice[function_rows - 1][j]
                else:
                    function_sum += function_lattice[i][j] * function_lattice[i][0] + function_lattice[i][j] * \
                                    function_lattice[i + 1][j] + function_lattice[i][j] * \
                                    function_lattice[i][j - 1] + function_lattice[i][j] * \
                                    function_lattice[function_rows - 1][j]
            else:
                if j == 0:
                    function_sum += function_lattice[i][j] * function_lattice[i][j + 1] + function_lattice[i][j] * \
                                    function_lattice[0][j] + function_lattice[i][j] * \
                                    function_lattice[i][function_columns - 1] + function_lattice[i][j] * \
                                    function_lattice[i - 1][j]
                elif j < function_columns - 1:
                    function_sum += function_lattice[i][j] * function_lattice[i][j + 1] + function_lattice[i][j] * \
                                    function_lattice[0][j] + function_lattice[i][j] * \
                                    function_lattice[i][j - 1] + function_lattice[i][j] * function_lattice[i - 1][j]
                else:
                    function_sum += function_lattice[i][j] * function_lattice[i][0] + function_lattice[i][j] * \
                                    function_lattice[0][j] + function_lattice[i][j] * \
                                    function_lattice[i][j - 1] + function_lattice[i][j] * function_lattice[i - 1][j]
    return function_sum / 2


@njit
def avg_spin(function_lattice):
    s = 0.0
    N = len(function_lattice) * len(function_lattice[0])
    for row in function_lattice:
        s += sum(row)
    return s / float(N)


@njit
def magnetization(function_lattice):
    s = 0.0
    for row in function_lattice:
        s += sum(row)
    return s


@njit
def monte_carlo_step(function_first_lattice, function_second_lattice, function_total_energy, function_temperature, lattice_size):
    function_first_lattice = function_first_lattice.copy()
    function_second_lattice = function_second_lattice.copy()
    for row in range(lattice_size):
        for column in range(lattice_size):
            change = function_first_lattice[row][column] * -1
            function_second_lattice[row][column] = change
            d_U = calculate_energy_difference(function_second_lattice, row, column, lattice_size, lattice_size)
            w = min([1, np.exp(d_U / function_temperature)])
            R2 = random.random()
            if w > R2:
                function_total_energy += d_U
                function_first_lattice = function_second_lattice.copy()
            else:
                function_second_lattice = function_first_lattice.copy()
    return function_first_lattice, function_total_energy


if __name__ == "__main__":
    sizes = [10, 40, 120]
    Ul = []
    MonteCarloSteps = int(1.3 * 10 ** 5)
    big_m = []
    capacity = []
    all_scaled_Y = []
    all_scaled_X = []
    for size in sizes:
        print("Calculating for size: {}".format(size))
        rows = size
        columns = size
        main_lattice = np.ones([size, size], dtype=np.int64)
        ml = []
        temps = []
        heat_cap = []
        main_energy = calculate_system_energy(main_lattice, function_rows=rows, function_columns=columns)
        scal_Y = []
        scal_X = []
        for T in np.linspace(0.000001, 4, 80):
            print("Calculating for temperature: {}".format(T))
            first_lattice = deepcopy(main_lattice)
            second_lattice = deepcopy(main_lattice)
            magnetizations = []
            avm = 0.0
            avm4 = 0.0
            avm2 = 0.0
            c = 0
            avU1 = 0.0
            avU2 = 0.0
            total_energy = deepcopy(main_energy)
            for msc in range(0, MonteCarloSteps):
                second_lattice = deepcopy(first_lattice)
                first_lattice, total_energy = monte_carlo_step(first_lattice, second_lattice, total_energy, T, size)
                if msc > 30000 and msc % 100 == 0:
                    m = avg_spin(first_lattice)
                    avm += abs(m)
                    avm4 += math.pow(magnetization(first_lattice), 4)
                    avm2 += math.pow(magnetization(first_lattice), 2)
                    magnetizations.append(m)
                    c += 1
                    avU1 += total_energy
                    avU2 += math.pow(total_energy, 2)
                    if (T == 0.5063299873417721 or T == 2.278481443037975 or T == 3.037974924050633) and msc == 90000:
                        fig, ax = plt.subplots()
                        cmp = colors.ListedColormap(['white', 'black'])
                        ax.pcolormesh(range(0, len(first_lattice)), range(0, len(first_lattice[0])), first_lattice,
                                      cmap=cmp, vmin=-1.0, vmax=1.0)
                        fig.set_size_inches(9.5, 9.5)
                        ax.set_ylabel('Rows')
                        ax.set_xlabel('Columns')
                        plt.title("Heatmap of Ising model for temperature {}".format(T))
                        plt.savefig(RESULTS_DIR /
                                    ('Heatmap_for_T=' + str(T) + '_time=' + str(msc) + "L= " + str(size) + '.png'))
            if T == 0.5063299873417721 or T == 2.278481443037975 or T == 3.037974924050633:
                fig, ax = plt.subplots()
                ax.plot(magnetizations, ".")
                ax.set_xlabel('Time')
                ax.set_ylabel('Magnetization m')
                ax.set_ylim([-1, 1])
                fig.set_size_inches(9.5, 9.5)
                plt.title("Magnetization for temperature {}".format(T))
                plt.savefig(RESULTS_DIR / ('Magnetization_for_T=' + str(T) + "L= " + str(size) + '.png'))
            temps.append(avm / c)
            scal_Y.append(math.log((avm / c) * math.pow(size, 1 / 8), math.e))
            scal_X.append(math.log(((abs(T - 2.26) / 2.26) * size), math.e))
            ml.append(1 - ((avm4 / c) / (3 * (math.pow((avm2 / c), 2)))))
            avU1 = avU1 / c
            avU2 = avU2 / c
            cap = (avU2 - math.pow(avU1, 2)) / (rows * columns * T * T)
            heat_cap.append(cap)
        all_scaled_Y.append(scal_Y)
        all_scaled_X.append(scal_X)
        Ul.append(ml)
        big_m.append(temps)
        capacity.append(heat_cap)

    # Additional plots
    fig, ax = plt.subplots()
    ax.set_ylim([0, 1])
    ax.plot(np.linspace(0.000001, 4, 80), Ul[0], '-.', label="L=10")
    ax.plot(np.linspace(0.000001, 4, 80), Ul[1], '-.', label="L=40")
    ax.plot(np.linspace(0.000001, 4, 80), Ul[2], '-.', label="L=120")
    ax.set_xlabel('Temperature')
    ax.set_ylabel('U')
    ax.legend(loc='lower left', fontsize='large')
    fig.set_size_inches(9.5, 9.5)
    plt.savefig(RESULTS_DIR / 'Bineder.png')

    fig, ax = plt.subplots()
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 4])
    ax.plot(np.linspace(0.000001, 4, 80), big_m[0], '-.', label="L=10")
    ax.plot(np.linspace(0.000001, 4, 80), big_m[1], '-.', label="L=40")
    ax.plot(np.linspace(0.000001, 4, 80), big_m[2], '-.', label="L=120")
    ax.set_xlabel('T')
    ax.set_ylabel('<m>')
    ax.legend(loc='lower left', fontsize='large')
    fig.set_size_inches(9.5, 9.5)
    plt.savefig(RESULTS_DIR / 'AVGMAG.png')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(np.linspace(0.000001, 4, 80), capacity[0], '-.', label="L=10")
    ax.plot(np.linspace(0.000001, 4, 80), capacity[1], '-.', label="L=40")
    ax.plot(np.linspace(0.000001, 4, 80), capacity[2], '-.', label="L=120")
    ax.set_xlabel('T')
    ax.set_ylabel('c')
    ax.legend(loc='upper right', fontsize='large')
    fig.set_size_inches(9.5, 9.5)
    plt.savefig(RESULTS_DIR / 'heatcap.png')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(all_scaled_X[0], all_scaled_Y[0], '.', label="L=10")
    ax.plot(all_scaled_X[1], all_scaled_Y[1], '.', label="L=40")
    ax.plot(all_scaled_X[2], all_scaled_Y[2], '.', label="L=120")
    ax.set_xlabel('ln[(|1-T/Tc|)L^(1/v)]')
    ax.set_ylabel('ln[mL^(B/v)]')
    ax.legend(loc='lower left', fontsize='large')
    fig.set_size_inches(9.5, 9.5)
    plt.savefig(RESULTS_DIR / 'scaled.png')
