import mySVM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy import optimize
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


plt.figure(figsize=(5,5))
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)

# create linearly separable data
n_samples   =   25
# create random points from that are defined as y > x to be assign with value -1
X1 = [-1.0 + 2*np.random.random() for i in range(n_samples)]
X1 = [np.array([i, i + np.random.random()]) for i in X1]
Y1 = [-1 for i in range(len(X1))]
# create random points from that are defined as y < x to be assign with value 1
X2 = [-1.0 + 2*np.random.random() for i in range(n_samples)]
X2 = [np.array([i, i - np.random.random()]) for i in X2]
Y2 = [1 for i in range(len(X2))]

# add some outliers (the points y > x have the value 1 and y < x -1)
n_outliers = 4
X1_o = [-1.0 + 2*np.random.random() for i in range(n_outliers)]
X1_o = [np.array([i, i - np.random.random()]) for i in X1_o]
Y1_o = [-1 for i in range(len(X1_o))]
X2_o = [-1.0 + 2*np.random.random() for i in range(n_outliers)]
X2_o = [np.array([i, i + np.random.random()]) for i in X2_o]
Y2_o = [-1 for i in range(len(X2_o))]

# plot the values
plt.plot(np.arange(-1.0, 1.0, 1e-3), np.arange(-1.0, 1.0, 1e-3), linestyle = '--')
plt.scatter(np.array(X1)[:,0], np.array(X1)[:,1],       color = 'red')
plt.scatter(np.array(X1_o)[:,0], np.array(X1_o)[:,1],   marker = 'x', color = 'red')
plt.scatter(np.array(X2)[:,0], np.array(X2)[:,1],       color = 'blue')
plt.scatter(np.array(X2_o)[:,0], np.array(X2_o)[:,1],   marker = 'x', color = 'blue')
plt.xlabel("x")
plt.ylabel("y")
'''
# concatenate all the data
X_train = np.array(X1 + X2 + X1_o + X2_o)
Y_train = np.array(Y1 + Y2 + Y1_o + Y2_o)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)
plt.show()
'''
X_1 = np.array([1,1])
Y_1 = -1
X_2 = np.array([-1,-1])
Y_2 = 1
X_3 = np.array([0.5,0.5])
Y_3 = -1
X_4 = np.array([-0.5,-0.5])
Y_4 = 1
X_5 = np.array([-0.3,-0.1])
Y_5 = 1
X_6 = np.array([0.1,0.3])
Y_6 = -1
X_train = np.array([X_1, X_2, X_3, X_4, X_5, X_6])
Y_train = np.array([Y_1, Y_2, Y_3, Y_4, Y_5, Y_6])

a = mySVM.mySVM(W= np.random.random(2), svm_type="hard")
a.optimizeAlphas(X=X_train, Y=Y_train, C=2.0)
supports = a.findSupportVectors(Y= Y_train, X= X_train)
a.newWeights(supports)
b = a.generateB(supports)
a.findTheBestB(b, Y_train,X_train)
y_pred = a.fit(X_train)
print(y_pred)
a.drawLines(Y_train, X_train, supports, y_pred)
print(a.getWidth())


#--------------------------------------------------------------------
n       = 500
decision_function = lambda x, a, b: b*np.sqrt(1 - x**2/a**2)
A       = 0.707
b       = 3.14
X       = -2.0 + 2 * 2 * np.random.random(n)
Y       = -4.0 + 2 * 4 * np.random.random(n)
X_m1    = []
X_1     = []
Y_train = []
X_train = []

# add classes
for i, x in enumerate(X):
    if x**2 / A**2 + Y[i]**2 / b**2 > 1:
        X_1.append(np.array([x, Y[i]]))
        Y_train.append(1)
    else:
        X_m1.append(np.array([x, Y[i]]))
        Y_train.append(0)
    X_train.append(np.array([x, Y[i]]))

n = 20
X = -2.0 + 2 * 2 * np.random.random(n)
Y = -4.0 + 2 * 4 * np.random.random(n)
# add outliers
for i, x in enumerate(X):
    if x**2 / A**2 + Y[i]**2 / b**2 > 1:
        X_m1.append(np.array([x, Y[i]]))
        Y_train.append(0)
    else:
        X_1.append(np.array([x, Y[i]]))
        Y_train.append(1)
    X_train.append(np.array([x, Y[i]]))
X_1     = np.array(X_1)
X_m1    = np.array(X_m1)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
#----------------------------------------------------------------
a = mySVM.mySVM(W= np.random.random(2), svm_type='kernel', kernelType='poly')
a.calculateKernel(X_train, k=2)
a.optimizeAlphas(X=X_train, Y=Y_train, C=2.0)
support = a.findSupportVectors(Y_train, X_train)
n       = 20
X_test  = np.array([np.array([-2.0 + 2 * 2 * np.random.random(), -4.0 + 2 * 4 * np.random.random()]) for i in range(n)])
Y_test = []
for x, y in X_test:
    if x**2 / A**2 + y**2 / b**2 > 1:
        Y_test.append(1)
    else:
        Y_test.append(0)
y_pred = a.fit(X_train, support)
print(Y_test)
print(y_pred)
print(f"Accuracy={accuracy_score(y_pred, Y_train):.2e}")

Ytest_pred = a.predict(X_test, support)
print(f"Accuracy={accuracy_score(Ytest_pred, Y_test):.2e}")
a.drawForKernel(Y_train, X_train, X_test, y_pred)
