import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import EllipseCollection

def plot_corr_ellipses(data, ax=None, **kwargs):

    M = np.array(data)
    if not M.ndim == 2:
        raise ValueError('data must be a 2D array')
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'aspect':'equal'})
        ax.set_xlim(-0.5, M.shape[1] - 0.5)
        ax.set_ylim(-0.5, M.shape[0] - 0.5)

    # xy locations of each ellipse center
    xy = np.indices(M.shape)[::-1].reshape(2, -1).T

    # set the relative sizes of the major/minor axes according to the strength of
    # the positive/negative correlation
    w = np.ones_like(M).ravel()
    h = 1 - np.abs(M).ravel()
    a = 45 * np.sign(M).ravel()

    ec = EllipseCollection(widths=w, heights=h, angles=a, units='x', offsets=xy,
                           transOffset=ax.transData, array=M.ravel(), **kwargs)
    ax.add_collection(ec)

    # if data is a DataFrame, use the row/column names as tick labels
    if isinstance(data, pd.DataFrame):
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(data.columns, rotation=90)
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_yticklabels(data.index)

    return ec
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

diabetes = pd.read_csv("diabetes.csv")
glucose = diabetes['Glucose']

print(diabetes.head())
print(glucose.quantile(q=[0.25, 0.5, 0.75]))

glucose.hist(bins=20)
plt.title("Histogram of glucose")
plt.xlabel("Glucose")
plt.show()

glucose.plot.kde()
plt.show()
diabetes['Age'].hist()
plt.title("Histogram of Age")
plt.xlabel("Age")
plt.show()
diabetes['Pregnancies'].hist()
plt.title("Histogram of pregnancies")
plt.xlabel("Pregnancies")
plt.show()

glucose.plot(kind='box', title="Box plot of glucose")
plt.show()

out1= diabetes.loc[diabetes['Outcome'] == 0]['Glucose']
out2 = diabetes.loc[diabetes['Outcome'] == 1]['Glucose']
data = [out1, out2]
plt.violinplot(data, showmedians=True)
plt.ylabel('Glucose')
plt.show()

print("Trimmed mean for 10% trim: ", sc.stats.trim_mean(glucose, 0.1))
print("Trimmed mean for 20% trim: ", sc.stats.trim_mean(glucose, 0.25))
print("Trimmed mean for 40% trim: ", sc.stats.trim_mean(glucose, 0.4))

print(sc.stats.mstats.trimmed_var(glucose, limits=(0.1, 0.1), inclusive=(1,1), relative=True, ddof=True))
print(sc.stats.mstats.trimmed_var(glucose, limits=(0.2, 0.2), inclusive=(1,1), relative=True, ddof=True))
print(sc.stats.mstats.trimmed_var(glucose, limits=(0.3, 0.3), inclusive=(1,1), relative=True, ddof=True))

def findLambda(data):
    l = []
    for x in data:
        l.append(-np.log(1- x))
    return sum(l)/len(data)

preg = diabetes["Pregnancies"]
data = np.histogram(preg)


data = data[0]/(data[0][0]+1)
lmbd = findLambda(data)
lmbdMLE = preg.mean()
print("Lambda is equal to: ", lmbd)
print("Lambda MLE is equal to: ", lmbdMLE)
print(data)

plt.title("Histogram of pregnancies")
plt.xlabel("Pregnancies")
s = np.random.poisson(lmbd, 50000)
count, bins, ignored = plt.hist(s, 20, density=True)
s = np.random.poisson(lmbdMLE, 20000)
count, bins, ignored = plt.hist(s, 20, density=True, color='r')
plt.scatter(range(0,10), data, c='green')
plt.show()

data = diabetes[['Glucose', 'Age', 'BMI', 'Pregnancies', 'DiabetesPedigreeFunction']]
corrMatrix = data.corr(method='pearson')
print(corrMatrix)
fig, ax = plt.subplots(1, 1)
m2 = plot_corr_ellipses(corrMatrix, ax=ax, cmap='seismic', clim=[-1, 1])
cb2 = fig.colorbar(m2)
ax.margins(0.3)
plt.show()
bmi = diabetes["BMI"]
plt.scatter(glucose, bmi)
plt.show()
plt.hexbin(glucose, bmi, gridsize=50)
plt.show()
