import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


class Automata:

    def __init__(self, size: int, time: int, rule_number: int, bc='periodic', point_dist='normal', number_of_ones=1):
        if point_dist == 'normal':
            lattice = np.zeros(size)
            lattice[int(size / 2)] = 1
            self.lattice = lattice
        elif point_dist == 'random':
            lattice = np.hstack((np.ones(number_of_ones), np.zeros(size - number_of_ones)))
            np.random.shuffle(lattice)
            self.lattice = lattice
        self.L = size
        self.BC = bc
        self.T = time
        self.ruleNumber = rule_number
        self.rules = self.get_binary_rule()
        self.result = None
        self.resultMatrix = None

    def get_binary_rule(self):
        binaryRule = "{0:08b}".format(self.ruleNumber)
        rules = list(binaryRule)
        rules.reverse()
        ones = []
        for i, r in enumerate(rules):
            if int(r) == 1:
                R = "{0:03b}".format(i)
                R = list(map(int, list(R)))
                ones.append(R)
        return ones

    def generate(self):
        resultMatrix = []
        if self.BC == 'periodic':
            resultMatrix.append(self.lattice)
            previousLattice = self.lattice
            for _ in range(self.T):
                next_lattice = [0] * self.L
                for i in range(self.L):
                    if i == 0:
                        joined = [previousLattice[i - 1]] + [previousLattice[i]] + [previousLattice[i + 1]]
                        if joined in self.rules:
                            next_lattice[i] = 1
                    elif i == self.L - 1:
                        joined = [previousLattice[i - 1]] + [previousLattice[i]] + [previousLattice[0]]
                        if joined in self.rules:
                            next_lattice[i] = 1
                    else:
                        joined = [previousLattice[i - 1]] + [previousLattice[i]] + [previousLattice[i + 1]]
                        if joined in self.rules:
                            next_lattice[i] = 1
                resultMatrix.append(next_lattice)
                previousLattice = next_lattice
        resultMatrix.reverse()
        self.resultMatrix = resultMatrix

    def generate_plot(self):
        fig, ax = plt.subplots()
        cmp = colors.ListedColormap(['white', 'black'])
        ax.pcolormesh(range(0, self.L), range(0, self.T + 1), self.resultMatrix, cmap=cmp, vmin=0, vmax=1,
                      edgecolors='gray', linewidths=0.1)
        # fig.set_size_inches(9.5, 9.5)
        ax.set_ylabel('T')
        ax.set_xlabel('X')
        plt.title("Automata")
        plt.show()


if __name__ == "__main__":
    aut = Automata(size=31, time=15, rule_number=61)
    aut.generate()
    aut.generate_plot()
    aut2 = Automata(size=31, time=15, rule_number=61, point_dist='random', number_of_ones=15)
    aut2.generate()
    aut2.generate_plot()
