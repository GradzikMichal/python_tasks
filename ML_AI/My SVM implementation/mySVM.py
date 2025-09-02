import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy import optimize


class mySVM():
    def __init__(self, W: np.array, svm_type: str, b=0, alphas=None, kernelType='None'):
        self.alphas = alphas
        self.W = W
        self.N = len(W)
        self.b = b
        self.svm_type = svm_type
        self.best_accuracy = 0.0
        self.kernelType = kernelType
        self.kernel = None

    ################################################
    def get_N(self):
        return self.N

    def get_b(self):
        return self.b

    def get_W(self):
        return self.W

    def get_type(self):
        return self.svm_type

    ##############################################
    def set_N(self, N):
        self.N = N
        self.W = np.zeros(N)

    def set_b(self, b):
        self.b = b

    def set_W(self, W: np.array):
        self.N = len(W)
        self.W = W

    ############################################################
    def __getitem__(self, key):
        return self.W[key]

    def __setitem__(self, key, value):
        self.W[key] = value

    def __getslice(self, i, j):
        return self.W[i:j]

    ###################################################
    def Lagrange(self, alphas: np.ndarray, Y: np.ndarray, X: np.ndarray):
        if self.kernelType != 'None':
            return -(np.sum(alphas) - 0.5 * np.sum(np.outer(alphas.T, alphas) * np.outer(Y.T, Y) * self.kernel))
        return -(np.sum(alphas) - 0.5 * np.sum(np.outer(alphas.T, alphas) * np.outer(Y.T, Y) * np.dot(X, X.T)))

    def optimizeAlphas(self, X: np.ndarray, Y: np.ndarray, C=0.0):
        if self.svm_type == "soft":
            constraints = [optimize.LinearConstraint(Y, 0.0, 0.0)]
            for i in range(len(Y)):
                A = np.zeros(len(Y))
                A[i] = 1
                cons = optimize.LinearConstraint(A, 0.0, C)
                constraints.append(cons)
            a = optimize.minimize(self.Lagrange,
                                  x0=0.05 * np.random.random(len(Y)),
                                  args=(Y, X),
                                  constraints=constraints,
                                  options={'maxiter': 200})
            self.alphas = a['x']
            return self.alphas
        elif self.svm_type == "hard" or self.svm_type == "kernel":
            constraints = [optimize.LinearConstraint(Y, 0.0, 0.0)]
            for i in range(len(Y)):
                A = np.zeros(len(Y))
                A[i] = 1
                cons = optimize.LinearConstraint(A, 0.0, 0.0)
                constraints.append(cons)
            opt = optimize.minimize(self.Lagrange,
                                    x0=np.ones(len(Y)),
                                    args=(Y, X),
                                    constraints=constraints)
            self.alphas = opt['x']
            return self.alphas

    def calculateKernel(self, X, Z=None, k=1.0, c=0.0):
        if Z is None:
            XX = np.dot(X, X.T)
            XX = (XX + c) ** k
            self.kernel = XX
        else:
            XX = np.dot(X, Z.T)
            XX = (XX + c) ** k

            self.kernel = XX

        # new_X = []
        # for x in X:
        #    new_X.append(np.array([x[0]**2, x[1]**2, x[0]*x[1]]))
        # return np.array(new_X)

    def findSupportVectors(self, Y: np.ndarray, X: np.ndarray, close_one=1e-10):
        supports = []
        for i in range(len(self.alphas)):
            if self.alphas[i] > close_one or self.alphas[i] < -close_one:
                supports.append(np.array([self.alphas[i], Y[i], X[i]], dtype=object))
        return np.array(supports)

    def newWeights(self, supports=None, X=None, Y=None):
        if self.kernel == None:
            W = 0.0
            for alpha_i, y, x in supports:
                W += np.dot(alpha_i * y, x)
            self.W = W

    def generateB(self, supports):
        b = np.zeros(len(supports))
        for i, (alpha_i, y, x) in enumerate(supports):
            b[i] = y - np.dot(self.W, x)
        return b

    def findTheBestB(self, b, Y, X):
        bestB = 0
        bestAccuracy = 0.0
        for b_i in range(len(b)):
            y_pred = []
            for x in X:
                if np.dot(self.W, x) + b[b_i] < 0:
                    y_pred.append(-1)
                else:
                    y_pred.append(1)
            acc = accuracy_score(y_pred, Y)
            if acc > bestAccuracy:
                bestAccuracy = acc
                bestB = b[b_i]
        self.b = bestB

    def fit(self, X, support=None):
        if self.kernelType == 'None':
            y_pred = []
            for x in X:
                if np.dot(self.W, x) + self.b < 0:
                    y_pred.append(-1)
                else:
                    y_pred.append(1)
            return y_pred
        else:
            Y = support[:, 1]
            alphas = support[:, 0]
            result = np.dot(alphas, Y * self.kernel)
            y_pred = []
            for res in result:
                if res > 0:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
            return y_pred

    def predict(self, X, support):
        Z = support[:, 2]
        Y = support[:, 1]
        alphas = support[:, 0]
        xx_res = []
        for x in X:
            XX = 0.0
            for a, z, y in zip(alphas ,Z, Y):
                XX += a*y*np.dot(x, z)
            xx_res.append((XX) ** 2)
        y_pred = []
        for res in xx_res:
            if res > 0:
                y_pred.append(1)
            else:
                y_pred.append(0)
        return y_pred

    def calculateSlacks(self, X, Y):
        slackVars = []
        for i, x in enumerate(X):
            slackVars.append(np.max((0, 1 - (np.dot(self.W, x) + self.b) * Y[i])))
        return slackVars

    def getWidth(self):
        return 2 / (np.sqrt(self.W[0] ** 2 + self.W[1] ** 2))

    def drawLines(self, Y, X, supports, y_pred):
        fig, ax = plt.subplots(figsize=(5, 5))
        if self.svm_type == "soft":
            c = -self.b / self.W[1]
            c1 = -(self.b - 1) / self.W[1]
            c2 = -(self.b + 1) / self.W[1]
            a = -self.W[0] / self.W[1]
            print("Crossing point c = ", c, "Tangent coefficient a =", a)

            # perpendicular function to W - this can be defined for any support vectors as well and is
            # a decision line
            fp = lambda x: a * x + c
            fp1 = lambda x: a * x + c1
            fp2 = lambda x: a * x + c2
        if self.svm_type == "hard":
            c = -self.b / self.W[1]
            a = -self.W[0] / self.W[1]
            print("Crossing point c = ", c, "Tangent coefficient a =", a)
            X_2 = False
            X_1 = False
            for alpha, y, x in supports:
                if y > 0:
                    X_2 = x
                    if type(X_1) == np.ndarray:
                        break
                else:
                    X_1 = x
                    if type(X_2) == np.ndarray:
                        break
            # perpendicular function to W
            fp = lambda x: a * x + c
            fp1 = lambda x: (1 - self.b - self.W[0] * x) / self.W[1]
            fp2 = lambda x: (-1 - self.b - self.W[0] * x) / self.W[1]
        # a lambda for parallel function to W (contains the vector W)
        f = lambda x: -1.0 / a * x + c

        # set xrange
        x = np.arange(-2, 2, 0.01)

        # plot decision function
        ax.plot(x,
                fp(x),
                color='green',
                alpha=1,
                linewidth=7,
                label=f'$f(x)$ decision line for $b=${self.b:.4f}')

        # first category (+1)
        ax.plot(x,
                fp1(x),
                linestyle=':',
                color='blue',
                alpha=0.3,
                linewidth=3,
                label=f'$f(x)$ lower width of the street')
        # fill the values below lines
        ax.fill_between(x, x.min(), fp1(x), color='blue', alpha=0.3)

        # second category (-1)
        ax.plot(x,
                fp2(x),
                linestyle=':',
                color='red',
                alpha=0.3,
                linewidth=3,
                label=f'$f(x)$ upper width of the street')
        ax.fill_between(x, fp2(x), x.max(), color='red', alpha=0.3)

        # plot the weights function
        ax.plot(x,
                f(x),
                color='black',
                alpha=0.3,
                linewidth=7,
                label=r'$\vec{W}$ for $b=$' + f'{self.b:.4f}')

        # plot W vector
        norm = np.sqrt(np.sum(np.square(self.W)))
        ax.arrow(0,
                 c,
                 self.W[0] / norm,
                 self.W[1] / norm,
                 head_width=0.1,
                 width=0.02,
                 head_length=0.1,
                 color='black')

        # plot labels
        ax.text(self.W[0] / norm + 0.1,
                self.W[1] / norm - 0.1,
                r'$\vec{W}$',
                fontsize=20)

        # to plot width
        # ax.text(0 - 1.8, c ,
        #        f'width={np.dot(Sone - Smone, W) / norm : .1e}',
        #        rotation = np.math.tanh(a) * 360 / 2 / np.pi)

        # some supports

        # scatter the points, mark the outliers with x
        # ax.scatter(np.array(X1)[:,0], np.array(X1)[:,1],        marker = "o",    color = 'red')
        # ax.scatter(np.array(X1_o)[:,0], np.array(X1_o)[:,1],    marker = "x",    color = 'red')
        # ax.scatter(np.array(X2)[:,0], np.array(X2)[:,1],        marker = "o",    color = 'blue')
        # ax.scatter(np.array(X2_o)[:,0], np.array(X2_o)[:,1],    marker = "x",    color = 'blue')

        # plot the support vectors

        ax.set_title(f"Accuracy={accuracy_score(y_pred, Y):.2e}")

        # mark the tests
        # colors = ['blue' if (x == 1) else 'red' for x in y_pred]
        # ax.scatter(np.array(X)[:, 0], np.array(X)[:, 1],
        #           marker="s", color=colors, label="Test variables")
        # for alpha, YY, XX in supports:
        #    c = 'red' if (YY == -1) else 'blue'
        #    ax.scatter(XX[0], XX[1], marker='x', color=c)
        print(Y)
        colors = ['blue' if (x == 1) else 'red' for x in Y]
        ax.scatter(np.array(X)[:, 0], np.array(X)[:, 1],
                   marker=".", color=colors, label="Train variables")

        ax.axhline(0, color='grey')
        ax.axvline(0, color='grey')
        ax.set_ylabel('$X_1$')
        ax.set_xlabel('$X_0$')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        # plt.legend(frameon=True)
        plt.show()

    def drawForKernel(self, Y, X, Y_pred_train,x_test, y_test, Y_pred_test):
        decision_function = lambda x, a, b: b * np.sqrt(1 - (x ** 2 / a ** 2))
        a = 0.707
        b = 3.14
        x = np.arange(-2.5, 2.5, 1e-4)
        plt.plot(x, decision_function(x, a, b), linestyle='--', color='black')
        plt.plot(x, -decision_function(x, a, b), linestyle='--', color='black')
        colors = ['red' if i == 0 else 'blue' for i in Y]
        zero, one = None, None
        for i in range(len(colors)):
            if colors[i] == 'red':
                zero = plt.scatter(X[i][0], X[i][1], color=colors[i], label='train -> 0')
            else:
                one = plt.scatter(X[i][0], X[i][1], color=colors[i], label='train -> 1')

        plt.legend(handles=[zero, one])

        plt.show()
