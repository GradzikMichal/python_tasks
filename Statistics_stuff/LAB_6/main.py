import random
import numpy
import scipy as sc
import numpy as np
import numpy.random as nprand
import matplotlib.pyplot as plt


def zad1():
    dice_roll = random.choices(population=[1, 2, 3, 4, 5, 6], weights=[1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 2 / 7], k=20)
    print(dice_roll)


def zad2():
    print(nprand.choice(range(5, 101), size=10, replace=False))


def zad3():
    print(sc.stats.binom(n=2000, p=0.5).mean())


def zad4():
    mu, sigma = 0, 1
    s_array = []
    for i in range(9):
        s_array.append(nprand.normal(mu, sigma, 10 ** i))
    fig, axs = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            axs[i][j].hist(s_array[i * 3 + j], 100, density=True)
    plt.show()


def zad5():
    p = 0.15
    n_trails = 2000
    r = sc.stats.bernoulli.rvs(p, size=n_trails)
    print("Number of successes: ", r.sum())
    length = 0
    k = 0
    for x in r:
        if x == 1:
            k += 1
        elif k != 0:
            if k > length:
                length = k
            k = 0
    print("Longest winning spree is: ", length)
    at_least_351 = sc.stats.binom.sf(k=351, n=n_trails, p=p)
    at_most_249 = sc.stats.binom.cdf(k=249, n=n_trails, p=p)
    print("Probability: ", at_most_249 + at_least_351)
    T = 1000
    for p_i in np.linspace(0.15, 1, 86):
        count = 0
        for i in range(T):
            r = sc.stats.bernoulli.rvs(p_i, size=n_trails)
            length = 0
            k = 0
            for x in r:
                if x == 1:
                    k += 1
                elif k != 0:
                    if k > length:
                        length = k
                    k = 0
            if length >= 30:
                count += 1
        if count > 0.75 * T:
            print("p needed: ", p_i)
            break


def zad6():
    n_tosses = 1000
    n_iter = 1000
    tosses = sc.stats.binom.rvs(n=n_tosses, p=0.5, size=n_iter)
    n_of_div_11 = (tosses % 11).tolist().count(0)
    print("probability is: ", n_of_div_11 / n_iter)


def zad7():
    mu, sigma = 177, 8
    n_iter = 10 ** 8
    normal = nprand.normal(mu, sigma, n_iter)
    prob_200 = np.where(normal > 200, 1, 0).sum() / n_iter
    print("Prob of a man higher than 200: ", prob_200)
    prob_160_180 = np.where(((160 <= normal) & (normal <= 180)), 1, 0).sum() / n_iter
    print("Prob of a man between 160-180: ", prob_160_180)
    bins = n_iter // 20
    probs_165 = []
    for i in range(bins):
        bin = normal[20*i:19+i*20]
        if np.where(bin > 165, 1, 0).sum() == 0:
            print("yey")
            probs_165.append(1)
        else:
            probs_165.append(0)
    prob_165 = float(sum(probs_165))/float(bins)
    print("Approx prob: ", prob_165)

def zad8():
    number = range(50, 101)
    n_iter = 2000
    V1 = np.array(random.choices(population=number, k=n_iter))
    normal_V2 = nprand.normal(0, np.sqrt(20), n_iter)
    V2 = V1 + normal_V2
    normal_V3 = nprand.normal(1, np.sqrt(100), n_iter)
    V3 = V1 + normal_V3
    V1_V1_coeff = np.corrcoef(V1, V1)
    print("Correlation between V1 and V1: ", V1_V1_coeff[0, 1])
    V1_V2_coeff = np.corrcoef(V1, V2)
    print("Correlation between V1 and V2: ", V1_V2_coeff[0, 1])
    V1_V3_coeff = np.corrcoef(V1, V3)
    print("Correlation between V1 and V3: ", V1_V3_coeff[0, 1])


if __name__ == "__main__":
    zad8()
