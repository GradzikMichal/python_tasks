import random
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange, njit


def zad1():
    g1 = lambda x: x
    h1 = lambda x: 100 * np.sqrt(x)
    h2 = lambda x: np.log(x)
    l1 = lambda x: np.sqrt(x)
    g2 = lambda x: x * np.sqrt(x)
    h3 = lambda x: x * np.log(x)
    l2 = lambda x: np.power(x, 2)
    g3 = lambda x: x / np.log(np.log(x))
    h4 = lambda x: x / np.log(x)
    g4 = lambda x: np.log(x) / np.log(np.log(x))
    h5 = lambda x: np.log(np.log(x)) / np.log(3)
    x1 = np.linspace(0, 500, 501)
    y_g1 = g1(x1)
    y_h1 = h1(x1)
    plt.plot(x1, y_g1, label='y = x')
    plt.plot(x1, y_h1, label='y = 100 * sqrt(x)')
    plt.legend()
    plt.show()
    x2 = np.linspace(0, 10000000, 10000001)
    y_g2 = g1(x2)
    y_h2 = h1(x2)
    plt.plot(x2, y_g2, label='y = x')
    plt.plot(x2, y_h2, label='y = 100 * sqrt(x)')
    plt.legend()
    plt.show()
    x3 = np.linspace(1, 10, 10)
    y_g3 = g1(x3)
    y_h3 = h2(x3)
    y_l3 = l1(x3)
    plt.plot(x3, y_g3, label='y = x')
    plt.plot(x3, y_h3, label='y = ln(x)')
    plt.plot(x3, y_l3, label='y = sqrt(x)')
    plt.legend()
    plt.show()
    x4 = np.linspace(1, 200, 200)
    y_g4 = g1(x4)
    y_h4 = h2(x4)
    y_l4 = l1(x4)
    plt.plot(x4, y_g4, label='y = x')
    plt.plot(x4, y_h4, label='y = ln(x)')
    plt.plot(x4, y_l4, label='y = sqrt(x)')
    plt.legend()
    plt.show()
    x5 = np.linspace(1, 50, 50)
    y_g5 = g2(x5)
    y_h5 = h3(x5)
    y_l5 = l2(x5)
    plt.plot(x5, y_g5, label='y = x * sqrt(x)')
    plt.plot(x5, y_h5, label='y = x * ln(x)')
    plt.plot(x5, y_l5, label='y = x * x')
    plt.legend()
    plt.show()
    x6 = np.linspace(1, 2000, 2000)
    y_g6 = g3(x6)
    y_h6 = h4(x6)
    plt.plot(x6, y_g6, label='y = x / ln(ln(x))')
    plt.plot(x6, y_h6, label='y = x / ln(x)')
    plt.legend()
    plt.show()
    x7 = np.linspace(1, 50000, 50001)
    y_g7 = g4(x7)
    y_h7 = h5(x7)
    plt.plot(x7, y_g7, label='y = ln(x) / ln(ln(x))')
    plt.plot(x7, y_h7, label='y = ln(ln(x)) / ln(3)')
    plt.legend()
    plt.show()


@jit
def zad2_bins():
    T = 1000
    arr_num_for_n = []
    for n in [10, 100, 1000]:
        arr_for_step = []
        for _ in range(T):
            bins = np.zeros(n)
            empty_bins = len(np.argwhere(bins))
            num_of_steps = 0
            rng = np.random.default_rng()
            while empty_bins != n:
                bin_num = rng.integers(low=0, high=n, size=1)
                bins[bin_num] += 1
                num_of_steps += 1
                empty_bins = len(np.argwhere(bins))
            arr_for_step.append(num_of_steps)
        arr_num_for_n.append(arr_for_step)
    return arr_num_for_n


@jit
def zad3_bins():
    T = 1000
    arr_num_for_n = []
    for n in [10, 100, 1000]:
        arr_for_step = []
        for _ in range(T):
            bins = np.zeros(n)
            empty_bins = 0
            num_of_steps = 0
            rng = np.random.default_rng()
            while empty_bins != 1:
                bin_num = rng.integers(low=0, high=n, size=1)
                bins[bin_num] += 1
                num_of_steps += 1
                empty_bins = len(np.argwhere(bins == 2))
            arr_for_step.append(num_of_steps)
        arr_num_for_n.append(arr_for_step)
    return arr_num_for_n


@jit
def zad4():
    T = 1000
    arr_num_for_n = []
    for m, n in [(1000, 1000), (100, 1000), (2000, 100)]:
        arr_for_step = []
        arr_of_numb_of_empty = []
        for _ in range(T):
            bins = np.zeros(n)
            rng = np.random.default_rng()
            for _ in range(m):
                bin_num = rng.integers(low=0, high=n, size=1)
                bins[bin_num] += 1
            arr_for_step.append(max(bins))
            arr_of_numb_of_empty.append(len(np.argwhere(bins == 0)))
        print("Average number of balls in the most loaded bin for m= " + str(m) + " and n= " + str(
            n) + " is equal to: " + str(float(sum(arr_for_step)) / float(T)))
        print("Average number of empty bins for m= " + str(m) + " and n= " + str(n) + " is equal to: " + str(
            float(sum(arr_of_numb_of_empty)) / float(T)))
        arr_num_for_n.append(sum(arr_for_step) / T)
    return arr_num_for_n


@njit(parallel=True, fastmath=True)
def approx_ec():
    T = 100
    ec = []
    for n in prange(10, 50001):
        arr_for_step = []
        for _ in prange(T):
            bins = np.zeros(n)
            num_of_steps = 0
            while (bins == 0).any():
                bin_num = random.randint(0, n - 1)
                bins[bin_num] += 1
                num_of_steps += 1
            arr_for_step.append(num_of_steps)
        ec.append(sum(arr_for_step) / T)
    return ec


def zad2():
    ec = approx_ec()
    plt.plot(range(10, 50001), ec)
    plt.show()
    arr = zad2_bins()
    plt.hist(arr[0], bins=range(0, max(arr[0]) + 2, 2))
    plt.show()
    plt.hist(arr[1], bins=range(0, max(arr[1]) + 20, 20))
    plt.show()
    plt.hist(arr[2], bins=range(0, max(arr[2]) + 200, 200))
    plt.show()


def zad3():
    arr = zad3_bins()
    plt.hist(arr[0], bins=10)
    plt.show()
    plt.hist(arr[1], bins=100)
    plt.show()
    plt.hist(arr[2], bins=1000)
    plt.show()


def biba_model(m, n, d, bins):
    rng = np.random.default_rng()
    bin_num = rng.integers(low=0, high=n, size=d)
    min_v = m + 1
    min_idx = 0
    for b in bin_num:
        b_v = bins[b]
        if b_v < min_v:
            min_v = b_v
            min_idx = b
    bins[min_idx] += 1
    return bins


if __name__ == '__main__':
    approx_ec()
