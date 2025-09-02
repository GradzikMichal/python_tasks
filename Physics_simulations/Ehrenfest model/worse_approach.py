import random
import matplotlib.pyplot as plt
import numpy
import math
from scipy.interpolate import CubicSpline
import scipy.special as sc

time = 10000
N = int(1000000)
p = 0.4
dogA = [i for i in range(1, N)]
dogB = []
fleadogA = []
fleadogB = []
fleadogA.append(len(dogA))
fleadogB.append(len(dogB))
chosenFlea = 7
chosenFleadog = []
chfleas = int(p * N)
for i in range(0, time):
    print(i)
    for j in range(1, chfleas):
        flea = random.randint(1, N - 1)
        jump = numpy.random.random_sample()
        print(j)
        if jump <= p:
            if flea in dogA:
                dogB.append(dogA.pop(dogA.index(flea)))
            else:
                dogA.append(dogB.pop(dogB.index(flea)))

    fleadogA.append(len(dogA))
    fleadogB.append(len(dogB))
    if chosenFlea in dogA:
        chosenFleadog.append(0)
    else:
        chosenFleadog.append(1)

fig, axs = plt.subplots(2, 1)
axs[0].plot(range(1, time), fleadogA, range(1, time), fleadogB)
axs[0].set_xlabel('Time [MSC]')
axs[0].set_ylabel('Number of fleas')
axs[0].grid(False)
axs[0].set_xlim(0, 200)
axs[1].plot(range(1, time - 1), chosenFleadog)
axs[1].set_xlim(0, 200)
fig.set_size_inches(10.5, 10.5)
fig.set_dpi(300)
plt.figtext(0.125, 0.96,
            "p: " + str(p) + "\n N: " + str(N) + "\n time: " + str(time),
            horizontalalignment="center",
            verticalalignment="top", weight='bold',
            wrap=False, fontsize=14,
            color="black", bbox={'facecolor': 'grey',
                                 'alpha': 0.9, 'pad': 5})
plt.savefig('Fleas_on_dogA_and_dogB_N=' + str(N) + '.png')
plt.show()

d = {}
for i in range(1, N):
    d[str(i)] = 0
for flea in fleadogA:
    if str(flea) in d:
        d[str(flea)] += 1

fig, axs = plt.subplots(1, 1)
axs.scatter(d.keys(), d.values() / time, c='grey', label='MC simulation')
axs.set_xlabel('Na')
axs.set_ylabel('Count')
plt.xticks(range(-1, N + 1, int(N / 10)))
fig.set_size_inches(10.5, 10.5)
fig.set_dpi(300)
plt.figtext(0.22, 0.96,
            "Area of histogram: " + str(round(s, 3)) + "\n p: " + str(p) + "\n N: " + str(N) + "\n time: " + str(time),
            horizontalalignment="center",
            verticalalignment="top", weight='bold',
            wrap=False, fontsize=14,
            color="black", bbox={'facecolor': 'grey',
                                 'alpha': 0.9, 'pad': 5})
# axs.plot(range(1,N), p_n, c="red", label='analytical')
axs.legend(loc='upper right', fontsize='large')
plt.savefig('Fleas_on_dogA_N=' + str(N) + 'P.png')
plt.show()
