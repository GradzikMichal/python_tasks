import json
import time
from math import log
from string import punctuation
from nltk.corpus import stopwords
from nltk.metrics import distance
from stemming.porter2 import stem
from nltk.book import text1, text2, text3
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans
import scipy.spatial as spatial
import matplotlib.pyplot as plt


def preprocessing8(text):
    words = list(word.translate(str.maketrans('', '', punctuation)) for line in text for word in line.split())
    stopword = set(stopwords.words('english'))
    words = [w.lower() for w in words if w not in stopword]
    words = [stem(word) for word in words]
    words = {word for word in words if len(word) < 8}
    return words


def unzip_nparray(nparray):
    ar = []
    for x in nparray:
        ar2 = []
        for y in x:
            ar2.append(y)
        ar.append(ar2)
    return ar


def checking_centroids(arr):
    check = len(arr[0])
    to_del = []
    if len(arr) > 1:
        for r in range(0, len(arr)):
            for c in range(r+1, len(arr)):
                i = 0
                for s in range(check):
                    if arr[r][s] in arr[c]:
                        i += 1
                    if i == check:
                        if c not in to_del:
                            to_del.append(c)
                        break
    to_del.reverse()
    for d in to_del:
        arr.pop(d)
    return arr


def minhash_bands(n, s):
    """
    Given a number of minhashes n and similarity parameter s,
    suggest parameters b and r such that signatures of two sets
    are considered "potentially similar" iff their Jaccard similarity
    is around s. Sometimes b*r doesn't equal to n but result is acceptable. (it is because of return Int)

    :param n: int
        The number of minhashes. Bigger than zero
    :param s: float
        The similarity parameter. Bigger than zero
    :return (b, r) tuple of ints
        The suggested parameters b and r.
    """
    b = 1
    min_difference = float('inf')
    opt_r = 1
    while b <= n:
        r = n // b
        cur_s = (1 / b) ** (1 / r)
        if abs(cur_s - s) < min_difference:
            min_difference = abs(cur_s - s)
            opt_r = r
        b += 1
    return (n // opt_r), opt_r


def kmean_compare_K():
    '''
            Showing graphs for kmeans clustering with different k using k-means++

            Args:
                No input arguments

            Returns:
                Nothing.
                The function is generating graphs of kmeans clustering
    '''
    data = np.array(json.load(open("data.json")))
    kmean_2 = KMeans(init="k-means++", n_clusters=2)
    kmean_2.fit(data)
    kmean_3 = KMeans(init="k-means++", n_clusters=3)
    kmean_3.fit(data)
    kmean_4 = KMeans(init="k-means++", n_clusters=4)
    kmean_4.fit(data)
    kmean_5 = KMeans(init="k-means++", n_clusters=5)
    kmean_5.fit(data)

    h = 0.02

    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx2, yy2 = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    xx3, yy3 = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    xx4, yy4 = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    xx5, yy5 = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z2 = kmean_2.predict(np.c_[xx2.ravel(), yy2.ravel()])
    Z2 = Z2.reshape(xx2.shape)
    Z3 = kmean_3.predict(np.c_[xx3.ravel(), yy3.ravel()])
    Z3 = Z3.reshape(xx3.shape)
    Z4 = kmean_4.predict(np.c_[xx4.ravel(), yy4.ravel()])
    Z4 = Z4.reshape(xx4.shape)
    Z5 = kmean_5.predict(np.c_[xx5.ravel(), yy5.ravel()])
    Z5 = Z5.reshape(xx5.shape)
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(
        Z2,
        interpolation="nearest",
        extent=(xx2.min(), xx2.max(), yy2.min(), yy2.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )
    ax[0, 1].imshow(
        Z3,
        interpolation="nearest",
        extent=(xx3.min(), xx3.max(), yy3.min(), yy3.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )
    ax[1, 0].imshow(
        Z4,
        interpolation="nearest",
        extent=(xx4.min(), xx4.max(), yy4.min(), yy4.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )
    ax[1, 1].imshow(
        Z5,
        interpolation="nearest",
        extent=(xx5.min(), xx5.max(), yy5.min(), yy5.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )
    ax[0, 0].plot(data[:, 0], data[:, 1], "k.", markersize=2)
    centroid_2 = kmean_2.cluster_centers_
    ax[0, 1].plot(data[:, 0], data[:, 1], "k.", markersize=2)
    centroid_3 = kmean_3.cluster_centers_
    ax[1, 0].plot(data[:, 0], data[:, 1], "k.", markersize=2)
    centroid_4 = kmean_4.cluster_centers_
    ax[1, 1].plot(data[:, 0], data[:, 1], "k.", markersize=2)
    centroid_5 = kmean_5.cluster_centers_
    ax[0, 0].scatter(
        centroid_2[:, 0],
        centroid_2[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    ax[0, 1].scatter(
        centroid_3[:, 0],
        centroid_3[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    ax[1, 0].scatter(
        centroid_4[:, 0],
        centroid_4[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    ax[1, 1].scatter(
        centroid_5[:, 0],
        centroid_5[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    ax.flat[0].set(title="k = 2")
    ax.flat[1].set(title="k = 3")
    ax.flat[2].set(title="k = 4")
    ax.flat[3].set(title="k = 5")
    plt.show()


def kmean_compare_type():
    '''
    Showing graphs for comparison between different type of kmeans clustering with different k using k-means++ or normal k-means algorithm.
    Also printing in console how many different centroids were generated for different K.

    Args:
        No input arguments

    Returns:
        Nothing.
        The function is generating graphs of kmeans clustering
    '''
    data = np.array(json.load(open("data.json")))
    kmean_2 = KMeans(init="k-means++", n_clusters=2, n_init=1)
    kmean_3 = KMeans(init="k-means++", n_clusters=3, n_init=1)
    kmean_4 = KMeans(init="k-means++", n_clusters=4, n_init=1)
    kmean_5 = KMeans(init="k-means++", n_clusters=5, n_init=1)
    cent2 = []
    cent3 = []
    cent4 = []
    cent5 = []
    for i in range(5):
        kmean_2.fit(data)
        c = unzip_nparray(kmean_2.cluster_centers_)
        if c not in cent2:
            cent2.append(c)
        kmean_3.fit(data)
        c = unzip_nparray(kmean_3.cluster_centers_)
        if c not in cent3:
            cent3.append(c)
        kmean_4.fit(data)
        c = unzip_nparray(kmean_4.cluster_centers_)
        if c not in cent4:
            cent4.append(c)
        kmean_5.fit(data)
        c = unzip_nparray(kmean_5.cluster_centers_)
        if c not in cent5:
            cent5.append(c)
    cent2 = checking_centroids(cent2)
    cent3 = checking_centroids(cent3)
    cent4 = checking_centroids(cent4)
    cent5 = checking_centroids(cent5)
    cents_over_loops = [
        len(cent2),
        len(cent3),
        len(cent4),
        len(cent5)
    ]
    print("Doing loops generated number of centroids (for k = 2,3,4,5): ", cents_over_loops[0], cents_over_loops[1], cents_over_loops[2], cents_over_loops[3])
    kmean_5.fit(data)
    kmean_2n = KMeans(init="random", n_clusters=2, n_init=1)
    kmean_3n = KMeans(init="random", n_clusters=3, n_init=1)
    kmean_4n = KMeans(init="random", n_clusters=4, n_init=1)
    kmean_5n = KMeans(init="random", n_clusters=5, n_init=1)
    cent2r = []
    cent3r = []
    cent4r = []
    cent5r = []
    for i in range(5):
        kmean_2n.fit(data)
        c = unzip_nparray(kmean_2n.cluster_centers_)
        if c not in cent2r:
            cent2r.append(c)
        kmean_3n.fit(data)
        c = unzip_nparray(kmean_3n.cluster_centers_)
        if c not in cent3r:
            cent3r.append(c)
        kmean_4n.fit(data)
        c = unzip_nparray(kmean_4n.cluster_centers_)
        if c not in cent4r:
            cent4r.append(c)
        kmean_5n.fit(data)
        c = unzip_nparray(kmean_5n.cluster_centers_)
        if c not in cent5r:
            cent5r.append(c)
    cent2r = checking_centroids(cent2r)
    cent3r = checking_centroids(cent3r)
    cent4r = checking_centroids(cent4r)
    cent5r = checking_centroids(cent5r)
    cents_over_loops = [
        len(cent2r),
        len(cent3r),
        len(cent4r),
        len(cent5r)
    ]
    print("Doing loops generated number of centroids (for k = 2,3,4,5): ", cents_over_loops[0], cents_over_loops[1], cents_over_loops[2], cents_over_loops[3])
    h = 0.02
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx2, yy2 = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    xx3, yy3 = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    xx4, yy4 = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    xx5, yy5 = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    xx2n, yy2n = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    xx3n, yy3n = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    xx4n, yy4n = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    xx5n, yy5n = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z2 = kmean_2.predict(np.c_[xx2.ravel(), yy2.ravel()])
    Z2 = Z2.reshape(xx2.shape)
    Z3 = kmean_3.predict(np.c_[xx3.ravel(), yy3.ravel()])
    Z3 = Z3.reshape(xx3.shape)
    Z4 = kmean_4.predict(np.c_[xx4.ravel(), yy4.ravel()])
    Z4 = Z4.reshape(xx4.shape)
    Z5 = kmean_5.predict(np.c_[xx5.ravel(), yy5.ravel()])
    Z5 = Z5.reshape(xx5.shape)

    Z2n = kmean_2n.predict(np.c_[xx2n.ravel(), yy2n.ravel()])
    Z2n = Z2n.reshape(xx2n.shape)
    Z3n = kmean_3n.predict(np.c_[xx3n.ravel(), yy3n.ravel()])
    Z3n = Z3n.reshape(xx3n.shape)
    Z4n = kmean_4n.predict(np.c_[xx4n.ravel(), yy4n.ravel()])
    Z4n = Z4n.reshape(xx4n.shape)
    Z5n = kmean_5n.predict(np.c_[xx5n.ravel(), yy5n.ravel()])
    Z5n = Z5n.reshape(xx5n.shape)
    fig, ax = plt.subplots(2, 4)
    ax.flat[0].set(ylabel="k-means ++")
    ax.flat[4].set(ylabel="vanilla k-means")
    ax.flat[0].set(title="k = 2")
    ax.flat[1].set(title="k = 3")
    ax.flat[2].set(title="k = 4")
    ax.flat[3].set(title="k = 5")
    ax[0, 0].imshow(
        Z2,
        interpolation="nearest",
        extent=(xx2.min(), xx2.max(), yy2.min(), yy2.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )
    ax[1, 0].imshow(
        Z2n,
        interpolation="nearest",
        extent=(xx2n.min(), xx2n.max(), yy2n.min(), yy2n.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )
    ax[0, 1].imshow(
        Z3,
        interpolation="nearest",
        extent=(xx3.min(), xx3.max(), yy3.min(), yy3.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )
    ax[1, 1].imshow(
        Z3n,
        interpolation="nearest",
        extent=(xx3n.min(), xx3n.max(), yy3n.min(), yy3n.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )
    ax[0, 2].imshow(
        Z4,
        interpolation="nearest",
        extent=(xx4.min(), xx4.max(), yy4.min(), yy4.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )
    ax[1, 2].imshow(
        Z4n,
        interpolation="nearest",
        extent=(xx4n.min(), xx4n.max(), yy4n.min(), yy4n.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )
    ax[0, 3].imshow(
        Z5,
        interpolation="nearest",
        extent=(xx5.min(), xx5.max(), yy5.min(), yy5.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )
    ax[1, 3].imshow(
        Z5n,
        interpolation="nearest",
        extent=(xx5n.min(), xx5n.max(), yy5n.min(), yy5n.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )
    ax[0, 0].plot(data[:, 0], data[:, 1], "k.", markersize=2)
    centroid_2 = kmean_2.cluster_centers_
    ax[0, 1].plot(data[:, 0], data[:, 1], "k.", markersize=2)
    centroid_3 = kmean_3.cluster_centers_
    ax[0, 2].plot(data[:, 0], data[:, 1], "k.", markersize=2)
    centroid_4 = kmean_4.cluster_centers_
    ax[0, 3].plot(data[:, 0], data[:, 1], "k.", markersize=2)
    centroid_5 = kmean_5.cluster_centers_

    ax[1, 0].plot(data[:, 0], data[:, 1], "k.", markersize=2)
    centroid_2n = kmean_2n.cluster_centers_
    ax[1, 1].plot(data[:, 0], data[:, 1], "k.", markersize=2)
    centroid_3n = kmean_3n.cluster_centers_
    ax[1, 2].plot(data[:, 0], data[:, 1], "k.", markersize=2)
    centroid_4n = kmean_4n.cluster_centers_
    ax[1, 3].plot(data[:, 0], data[:, 1], "k.", markersize=2)
    centroid_5n = kmean_5n.cluster_centers_
    ax[0, 0].scatter(
        centroid_2[:, 0],
        centroid_2[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    ax[1, 0].scatter(
        centroid_2n[:, 0],
        centroid_2n[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    ax[0, 1].scatter(
        centroid_3[:, 0],
        centroid_3[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    ax[1, 1].scatter(
        centroid_3n[:, 0],
        centroid_3n[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    ax[0, 2].scatter(
        centroid_4[:, 0],
        centroid_4[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    ax[1, 2].scatter(
        centroid_4n[:, 0],
        centroid_4n[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    ax[0, 3].scatter(
        centroid_5[:, 0],
        centroid_5[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    ax[1, 3].scatter(
        centroid_5n[:, 0],
        centroid_5n[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def kmean_words():
    '''
        Printing 3 clusters of a words, created using k-means++. Whole algorithm calculate them in 30 minutes because of number of words. For lower number of words it will be faster. Below result:
        k=  0
        ['driven', 'dogger', 'edtion', 'besmok', 'manmak', 'follow', 'scotch', 'dearer', 'firmam', 'herald', 'indian', 'retard', 'reclus', 'amalek', 'theori', 'grizzl', 'strain', 'wallen', 'xxxix', 'jabbok', 'tarsus', 'pippin', 'waller', 'vidocq', 'stroke', 'apollo', 'garter', 'gravel', 'summit', 'hearth', 'resign', 'liveli', 'abroad', 'superb', 'chamoi', 'secret', 'leyden', 'lotion', 'elinor', 'weasel', 'covert', 'verili', 'violat', 'premis', 'reprob', 'dimens', 'primer', 'econom', 'vortic', 'passer', 'impuls', 'garmen', 'becalm', 'devote', 'mental', 'regain', 'broker', 'ramifi', 'wittic', 'miguel', 'improv', 'letter', 'recent', 'bestow', 'morrel', 'gazett', 'repast', 'caught', 'infest', 'cessat', 'discov', 'obviat', 'limpid', 'casket', 'promis', 'silent', 'sporti', 'boomer', 'pillag', 'doubli', 'rarmai', 'unjust', 'steadi', 'cherri', 'temper', 'memoir', 'belong', 'myself', 'exclud', 'stoker', 'verdur', 'behalf', 'resist', 'church', 'explor', 'patent', 'market', 'kedger', 'natura', 'cascad', 'affirm', 'experi', 'toward', 'spigot', 'extend', 'aggreg', 'frigid', 'hydrus', 'creami', 'damocl', 'giraff', 'epicur', 'huswif', 'strike', 'nobler', 'bombay', 'midian', 'emphat', 'stiver', 'weaver', 'massiv', 'shuttl', 'tistig', 'aileth', 'stilli', 'purest', 'weaken', 'monger', 'bunyan', 'patron', 'benign', 'undevi', 'untast', 'insurg', 'naaman', 'vaniti', 'furrow', 'negoti', 'pulpit', 'preach', 'cudgel', 'aboard', 'shebah', 'diffid', 'pincer', 'bewild', 'taught', 'miseri', 'hamper', 'firstl', 'endang', 'soland', 'exhum', 'unshor', 'muppim', 'busili', 'mexico', 'throng', 'bremen', 'sealin', 'tartar', 'malici', 'prairi', 'pensiv', 'hearti', 'easili', 'sadder', 'cultiv', 'impedi', 'explan', 'christ', 'remors', 'forego', 'celebr', 'mirror', 'besoot', 'prelud', 'outlaw', 'avatar', 'parcel', 'island', 'unwink', 'thwack', 'utmost', 'radiat', 'gaieti', 'combat', 'arrang', 'banner', 'intrud', 'rumour', 'mother', 'presto', 'signet', 'ascend', 'canari', 'reader', 'louder', 'ishuah', 'sphynx', 'nineti', 'purcha', 'ungrat', 'andrew', 'insert', 'inclin', 'nought', 'vulgar', 'bozrah', 'arctic', 'plough', 'realiz', 'ungird', 'silken', 'victim', 'deadli', 'whiten', 'confin', 'stress', 'stumbl', 'primit', 'forest', 'dagger', 'horror', 'caress', 'vindic', 'quohog', 'zeboim', 'gopher', 'pauper', 'apolog', 'fulfil', 'urgenc', 'fought', 'folder', 'within', 'starri', 'maltes', 'unknow', 'overse', 'pervad', 'hunter', 'jaalam', 'twinkl', 'villan', 'provid', 'destin', 'taylor', 'enumer', 'waterg', 'disrat', 'antler', 'plural', 'deaden', 'intric', 'hebrew', 'swindl', 'marvel', 'reject', 'wholli', 'cheeri', 'brandi', 'mantel', 'welter', 'limees', 'magnif', 'rotten', 'unstak', 'delect', 'casual', 'forgat', 'hanoch', 'fright', 'longer', 'typifi', 'upward', 'hazard', 'remedi', 'ventil', 'coolli', 'corner', 'zoolog', 'wateri', 'divulg', 'vernal', 'gnomon', 'fetcht', 'sleepi', 'civita', 'oblivi', 'bearer', 'porter', 'happen', 'lancet', 'hyster', 'inquir', 'overaw', 'specul', 'martin', 'prayer', 'pepper', 'shrine', 'maketh', 'wealth', 'thrill', 'vitiat', 'shriek', 'gainst', 'limber', 'mistak', 'finger', 'condit', 'requit', 'chines', 'unborn', 'materi', 'medica', 'unceas', 'potato', 'artedi', 'violin', 'feroci', 'beheld', 'credul', 'bright', 'breath', 'lucian', 'effect', 'reydan', 'crippl', 'absent', 'ponder', 'bounti', 'attest', 'larger', 'aright', 'result', 'sogger', 'rejoic', 'slowli', 'nurtur', 'intend', 'loveli', 'morass', 'begrim', 'digest', 'becaus', 'preced', 'partak', 'quiver', 'squint', '275th', 'borean', 'banter', 'cognac', 'parlor', 'joiner', 'profus', 'inmost', 'invest', 'simeon', 'claret', 'kemuel', 'reumah', 'yonder', 'placid', 'spermi', 'appeal', 'occup', 'birsha', 'tavern', 'oyster', 'clever', 'balien', 'astral', 'rather', 'enough', 'simper', 'tantal', 'tassel', 'london', 'decept', 'ensign', 'trepid', 'caudam', 'acquir', 'eighti', 'subtil', 'partit', 'gradat', 'poster', 'cinder', 'barrow', 'forbid', 'reform', 'scrawl', 'neglig', 'dreari', 'fellow', 'mibsam', 'shinar', 'cooper', 'bitter', 'earthi', 'cannib', 'grocer', 'crieth', 'archer', 'dollar', 'gutter', 'siddim', 'flavor', 'lizard', 'unholi', 'putrid', 'brutal', 'cotton', 'disagr', 'holder', 'atlant', 'return', 'troubl', 'inelig', 'clergi', 'jachin', 'tender', 'insult', 'barter', 'inclos', 'gilder', 'consid', 'offici', 'effus', 'byward', 'maladi', 'coenti', 'design', 'formal', 'kitten', 'falcon', 'eventu', 'revuls', 'verifi', 'matern', 'wetter', 'wonder', 'unfold', 'upcast', 'tinder', 'vestur', 'solado', 'relent', 'endwis', 'impati', 'awaken', 'seclud', 'crunch', 'clover', 'harden', 'penetr', 'stride', 'finest', 'obliqu', 'zaavan', 'intang', 'whalin', 'grisli', 'conced', 'boyish', 'liquor', 'shemeb', 'moment', 'astray', 'edmund', 'orient', 'mortem', 'unsolv', 'defend', 'braver', 'decant', 'harold', 'marrow', 'launch', 'beyond', 'dealer', 'ferrul', 'despit', 'embodi', 'chuckl', 'chisel', 'levant', 'forese', 'allegi', 'border', 'armada', 'inflex', 'fatten', 'capric', 'scoria', 'outcri', 'talent', 'debtor', 'cellar', 'sordid', 'insati', 'radney', 'disord', 'midwif', 'lavish', 'illaud', 'squash', 'dishon', 'answer', 'adjust', 'pascal', 'ablest', 'deprec', 'second', 'narrow', 'energi', 'unrifl', 'immedi', 'reckon', 'bishop', 'despis', 'vultur', 'sorter', 'equiti', 'invalu', 'immers', 'strand', 'pallor', 'carpet', 'aromat', 'thrice', 'giveth', 'switch', 'packag', 'analys', 'humbug', 'unread', 'scienc', 'harlot', 'comest', 'modifi', 'huppim', 'justic', 'faster', 'musket', 'averag', 'perfum', 'altern', 'bereav', 'pyrrho', 'revert', 'miracl', 'bestir', 'dribbl', 'spring', 'hermit', 'warmth', 'shepho', 'wooden', 'clinch', 'exclus', 'bolder', 'kittim', 'faulti', 'legaci', 'chalic', 'pocket', 'shuffl', 'asylum', 'improp', 'indign', 'plazza', 'sussex', 'amelia', 'sinist', 'quahog', 'attenu', 'strong', 'barton', 'mundan', 'deliri', 'octher', 'review', 'tandem', 'unwarr', 'servan', 'belfri', 'demesn', 'exhort', 'degrad', 'linger', 'meadow', 'patrol', 'heifer', 'incuri', 'edomit', 'formid', 'ribbon', 'racket', 'calneh', 'remain', 'devour', 'oregon', 'persia', 'easier', 'aspers', 'tremul', 'revolv', 'jimnah', 'capabl', 'trough', 'bonomi', 'bespok', 'darkey', 'beaten', 'occult', 'baleen', 'mortif', 'sullen', 'gossip', 'wearer', 'affair', 'angelo', 'lesson', 'howsev', 'procur', 'bowlin', 'captiv', 'roanok', 'lightn', 'turban', 'concur', 'honour', 'suffus', 'sahara', 'ignobl', 'outrag', 'vehicl', 'lentil', 'bleach', 'shalem', 'calmer', 'billet', 'suffoc', 'former', 'prefac', 'forgav', 'offens', 'rejoin', 'appear', 'muslin', 'chilli', 'albert', 'fossil', 'circul', 'misgiv', 'access', 'askanc', 'blanco', 'courag', 'differ', 'safeti', 'zillah', 'volunt', 'sceptr', 'worthi', 'quarto', 'wedder', 'cupola', 'immens', 'episod', 'binder', 'onward', 'jimimi', 'respir', 'stroll', 'unrest', 'mishap', 'zibeon', 'engend', 'skirra', 'pictur', 'amorit', 'mallet', 'foster', 'modest', 'hercul', 'shovel', 'accord', 'hospit', 'number', 'unlett', 'becket', 'grappl', 'shield', 'achbor', 'asshur', 'sebond', 'advent', 'vertic', 'wreath', 'append', 'father', 'gavest', 'condor', 'stowag', 'folger', 'seizur', 'though', 'moreov', 'physet', 'daniel', 'tortur', 'sooner', 'warden', 'origin', 'bloodi', 'erskin', 'sunday', 'requin', 'suicid', 'verbal', 'expans', 'dismal', 'marius', 'fertil', 'presag', 'cologn', 'parish', 'tragic', 'thrive', 'format', 'raptur', 'curtsi', 'hemdan', 'stream', 'dissip', 'pilfer', 'resolv', 'balanc', 'anomal', 'famish', 'shower', 'permit', 'upheld', 'wicket', 'lunaci', 'cambys', 'advers', 'climat', 'cloven', 'unskil', 'capsiz', 'unfort', 'holier', 'nutmeg', 'fasten', 'treadl', 'merman', 'dainti', 'nobodi', 'kelson', 'divers', 'abound', 'passag', 'sought', 'bashaw', 'lugger', 'master', 'german', 'incens', 'nearer', 'parson', 'crouch', 'situat', 'stubbl', 'potter', 'messag', 'sallow', 'forger', 'notori', 'extort', 'collar', 'burrow', 'provis', 'molten', 'ardour', 'virgin', 'presum', 'bildad', 'stimul', 'strake', 'detest', 'sorrow', 'judici', 'someth', 'strait', 'disint', 'foundl', 'acacia', 'saloon', 'eatabl', 'sprawl', 'amiabl', 'dispos', 'untold', 'reptil', 'caesar', 'legend', 'aghast', 'imperi', 'priori', 'villag', 'killer', 'poison', 'pinion', 'unpoet', 'comput', 'bilhan', 'streak', 'nipper', 'zimran', 'reseat', 'accept', 'hebron', 'heaven', 'religi', 'excus', 'enjoin', 'furnac', 'enforc', 'anymor', 'anatom', 'cancer', 'render', 'orphan', 'imprud', 'sheath', 'speedi', 'afraid', 'oblong', 'string', 'relaps', 'lazili', 'sprung', 'privat', 'absorb', 'season', 'trover', 'spirit', 'retort', 'citron', 'loveth', 'hilari', 'genius', 'vestig', 'diverg', 'reluct', 'embitt', 'flaxen', 'theatr', 'termin', 'sephar', 'quoggi', 'vicari', 'zodiac', 'huggin', 'liveth', 'senior', 'propel', 'spiral', 'deserv', 'disabl', 'profan', 'ceylon', 'unanim', 'assist', 'harbor', 'apport', 'bumper', 'tunnel', 'cymbal', 'spangl', 'weepon', 'expati', 'daught', 'chosen', 'margin', 'curaci', 'outdon', 'recept', 'interv', 'bailer', 'unreck', 'ardent', 'anamim', 'unawar', 'bedarn', 'martha', 'erudit', 'clammi', 'ladder', 'reiter', 'gether', 'adjoin', 'clamor', 'morton', 'column', 'phuvah', 'barrel', 'rapaci', 'unison', 'wampum', 'across', 'astern', 'regent', 'misnam', 'tether', 'paltri', 'clootz', 'walrus', 'detain', 'socket', 'offeri', 'twenti', 'unload', 'darker', 'philip', 'genial', 'tastin', 'mimick', 'redund', 'suffer', 'sermon', 'untidi', 'playth', 'ithran', 'luxuri', 'teenth', 'perhap', 'frosti', 'dugong', 'jemuel', 'tabret', 'pendul', 'uncork', 'loneli', 'undash', 'corpul', 'affect', 'beauti', 'afresh', 'torrid', 'litter', 'sherri', 'morbid', 'stolid', 'antiqu', 'saturn', 'concis', 'compet', 'yellow', 'fallen', 'desist', 'casino', 'prefer', 'becher', 'cardin', 'beacon', 'crappo', 'associ', 'besieg', 'reserv', 'festiv', 'pestil', 'jabber', 'trumpa', 'tumult', 'purpos', 'whilst', 'servic', 'sagaci', 'ravish', 'unmann', 'herman', 'primev', 'gratif', 'voraci', 'unheal', 'header', 'planet', 'corpor', 'signer', 'sunris', 'uplift', 'adieus', 'perish', 'insinu', 'scorch', 'unhint', 'includ', 'movabl', 'pigeon', 'mortal', 'darien', 'garden', 'climax', 'system', 'dietet', 'lessen', 'squeez', 'remind', 'sincer', 'crater', 'wherea', 'beguil', 'untott', 'auspic', 'convey', 'extant', 'habitu', 'thwart', 'hygien', 'aspect', 'undrap', 'ventur', 'israel', 'annuit', 'hurler', 'unmind', 'phrase', 'realli', 'wisher', 'veraci', 'zilpah', 'gather', 'action', 'wander', 'author', 'flinti', 'hawser', 'surfac', 'humili', 'entomb', 'regret', 'punish', 'rummag', 'deform', 'postur', 'duplic', 'vanish', 'bowstr', 'epitom', 'captor', 'maggot', 'crafti', 'milcah', 'garret', 'ephron', 'loosen', 'repuls', 'magian', 'motley', 'health', 'snatch', 'dictat', 'midway', 'provok', 'sultan', 'maiden', 'promot', 'drawer', 'strata', 'ridden', 'inclus', 'rotund', 'fabric', 'gunwal', 'newton', 'forbad', 'reprov', 'coffer', 'insens', 'famous', 'chalde', 'stigma', 'cyclad', 'nowher', 'infern', 'pedest', 'diseas', 'slight', 'samuel', 'inanim', 'indebt', 'cutter', 'abidah', 'instig', 'burial', 'scrape', 'bantam', 'cainan', 'anyway', 'tendin', 'relief', 'beaver', 'wrestl', 'sequel', 'saidst', 'arbour', 'seller', 'cousin', 'desecr', 'surviv', 'sinker', 'eclipt', 'yarman', 'martyr', 'deplor', 'africa', 'glimps', 'bushel', 'intern', 'spicin', 'seamen', 'hither', 'uphold', 'minded', 'better', 'encamp', 'wintri', 'tradit', 'outrid', 'suppos', 'shadow', 'observ', 'shrill', 'merari', 'shaker', 'repaid', 'gallop', 'appris', 'moabit', 'scuffl', 'jargon', 'deceiv', 'eldest', 'cabaco', 'repeat', 'rafter', 'eshban', 'gibson', 'sphinx', 'tendon', 'variat', 'barbar', 'rabbin', 'carson', 'joyous', 'knobbi', 'bedsid', 'explod', 'guttur', 'brazil', 'forgot', 'stolen', 'stript', 'lovest', 'hollow', 'callao', 'gambol', 'mibzar', 'extern', 'cobweb', 'patern', 'fissur', 'baltic', 'tearin', 'closer', 'portug', 'unpiti', 'admitt', 'inlaid', 'unbodi', 'vapour', 'berlin', 'redder', 'reclin', 'ungain', 'inevit', 'mutter', 'abrupt', 'dipper', 'muster', 'firmer', 'ostens', 'cleric', 'driver', 'savour', 'coward', 'inward', 'inform', 'declin', 'labori', 'carnat', 'wanton', 'dismay', 'nation', 'horner', 'passiv', 'whitak', 'pallet', 'cretan', 'necess', 'hittit', 'nescio', 'abhorr', 'gigant', 'statur', 'hanger', 'scroug', 'evolut', 'badger', 'niceti', 'beaker', 'modern', 'greedi', 'handed', 'primal', 'shrink', 'splash', 'spavin', 'siames', 'molest', 'ferret', 'shaggi', 'brazen', 'availl', 'extens', 'mouldi', 'sellin', 'elabor', 'enlist', 'compar', 'marten', 'beehiv', 'alpaca', 'govern', 'welfar', 'comedi', 'around', 'helena', 'kraken', 'poland', 'cypher', 'ascrib', 'comber', 'sister', 'wallow', 'devout', 'danger', 'applic', 'cavern', 'powder', 'butler', 'incred', 'vishnu', 'diffus', 'redeem', 'sublim', 'beckon', 'gambog', 'supern', 'injuri', 'motion', 'gulfwe', 'perman', 'recumb', 'featur', 'flurri', 'survey', 'indulg', 'sartin', 'prompt', 'majest', 'invinc', 'inshor', 'ferrar', 'thistl', 'retain', 'vortex', 'hasten', 'ledger', 'electr', 'arrowi', 'dishan', 'evapor', 'beggar', 'nimrod', 'oddest', 'canker', 'outrun', 'random', 'solemn', 'comrad', 'walnut', 'machin', 'throat', 'jordan', 'farmer', 'romant', 'proteg', 'sachem', 'mizpah', 'shrewd', 'shekel', 'conjur', 'routin', 'sawest', 'murray', 'defenc', 'crusad', 'repair', 'fleeci', 'vision', 'tophet', 'encumb', 'vacuum', 'dislik', 'deciph', 'orlean', 'naught', 'histor', 'kedron', 'wrapal', 'millin', 'behoov', 'profil', 'borneo', 'mechan', 'twould', 'bridal', 'galley', 'rubber', 'bamboo', 'deject', 'object', 'trampl', 'obscur', 'scarri', 'expert', 'vacant', 'beginn', 'sunset', 'etheri', 'halter', 'symbol', 'cheran', 'batter', 'dinner', 'mystic', 'rariti', 'unmean', 'liabil', 'baggag', 'widest', 'scanti', 'vexati', 'sailor', 'cogent', 'annual', 'geolog', 'pencil', 'carlin', 'unfurl', 'chapel', 'seldom', 'hamstr', 'infanc', 'asketh', 'winsom', 'woeful', 'stormi', 'brahma', 'scuttl', 'expens', 'cleans', 'intact', 'ecstat', 'latent', 'expend', 'arioch', 'auster', 'pompey', 'branch', 'morrow', 'errand', 'unsati', 'shiloh', 'breast', 'gestur', 'smithi', 'warlik', 'brittl', 'smooth', 'wisdom', 'crumpl', 'unwind', 'assert', 'pharez', 'romish', 'castor', 'sugari', 'puriti', 'dothan', 'breach', 'soloma', 'imagin', 'wretch', 'unmoor', 'wicked', 'hearer', 'unharm', 'childr', 'forbor', 'summon', 'hamlet', 'slower', 'camest', 'advanc', 'rocket', 'fleshi', 'turbid', 'occupi', 'unbent', 'multum', 'chezib', 'reveri', 'spleen', 'spectr', 'embark', 'doctor', 'anoint', 'albino', 'thrust', 'integu', 'kohath', 'poetic', 'legisl', 'defect', 'turbul', 'sperma', 'sureti', 'abomin', 'moieti', 'monkey', 'turkey', 'desert', 'french', 'balein', 'weston', 'spread', 'outfit', 'wherev', 'tallow', 'whimsi', 'attach', 'vendom', 'stoven', 'pantri', 'detach', 'disput', 'sarmon', 'durabl', 'enmiti', 'school', 'warmer', 'behead', 'amidst', 'distil', 'mastic', 'urgent', 'diabol', 'diklah', 'colour', 'latest', 'clayey', 'overdr', 'sphere', 'westwa', 'abstin', 'shambl', 'enfold', 'molifi', 'shingl', 'blessi', 'cordag', 'veriti', 'billow', 'inexor', 'ingeni', 'bouton', 'calcul', 'graven', 'button', 'canada', 'inland', 'phallu', 'pottag', 'surest', 'enlarg', 'timnah', 'fortif', 'imposs', 'revenu', 'confus', 'compli', 'unwilt', 'murder', 'evilli', 'foetal', 'airley', 'stupid', 'statut', 'liquid', 'odious', 'eschew', 'provoc', 'goblet', 'servil', 'palest', 'lawyer', 'fatigu', 'gotten', 'bigger', 'cucumb', 'unstir', 'fadest', 'burton', 'reliev', 'should', 'approb', 'unsuit', 'outyel', 'depriv', 'befool', 'carrol', 'moidor', 'cavali', 'pennon', 'hezron', 'nelson', 'respit', 'detail', 'vigour', 'actium', 'chasse', 'despot', 'deceas', 'compil', 'ratifi', 'candid', 'hallow', 'uproar', 'withal', 'adamit', 'robber', 'deafen', 'lather', 'undeni', 'friday', 'madest', 'divert', 'thirti', 'wither', 'decenc', 'zephyr', 'victor', 'disarm', 'potenc', 'dexter', 'nicest', 'austen', 'matter', 'height', 'artist', 'madden', 'closet', 'unfalt', 'smoker', 'kadesh', 'ishbak', 'cuvier', 'prelus', 'learnt', 'temani', 'vessel', 'arrest', 'strewn', 'mildew', 'inspir', 'mitten', 'nephew', 'syrian', 'pulver', 'frugal', 'wallet', 'vivaci', 'gibber', 'absenc', 'affabl', 'outlin', 'rachel', 'outran', 'consum', 'coffin', 'suburb', 'entitl', 'torpid', 'audibl', 'demand', 'golden', 'dunder', 'lowest', 'untrac', 'cranni', 'memori', 'prepar', 'sultri', 'infuri', 'uprais', 'europa', 'lubber', 'miller', 'reanim', 'chicha', 'locker', 'listen', 'butter', 'persev', 'easter', 'dryden', 'retent', 'suffic', 'compos', 'admiss', 'improb', 'teeter', 'marque', 'excurs', 'detect', 'oldest', 'shaken', 'friend', 'sabtah', 'empris', 'condol', 'lamech', 'bristl', 'poncho', 'pillar', 'suprem', 'actest', 'credit', 'migrat', 'canopi', 'asswag', 'essenc', 'measur', 'gilead', 'exuber', 'appeas', 'scroll', 'assuag', 'arrant', 'produc', 'joktan', 'makest', 'molass', 'normal', 'emblem', 'releas', 'forgiv', 'infatu', 'humour', 'censur', 'coloss', 'joseph', 'trophi', 'engrav', 'bought', 'period', 'infect', 'piazza', 'deceit', 'fairer', 'elucid', 'genesi', 'adbeel', 'expedi', 'raamah', 'siskur', 'grassi', 'summer', 'scrimp', 'upheav', 'darken', 'spinal', 'flight', 'behind', 'examin', 'durand', 'adroop', 'belial', 'stingi', 'reveal', 'instal', 'visual', 'hunger', 'intens', 'cultur', 'occurr', 'autumn', 'rugged', 'bellow', 'inculc', 'pretti', 'larder', 'firkin', 'penuri', 'digger', 'spoken', 'epidem', 'hussey', 'falter', 'goshen', 'assail', 'depict', 'cohort', 'saucer', 'common', 'unlimb', 'ginger', 'defray', 'deepli', 'merest', 'gospel', 'quaint', 'writer', 'ordain', 'miasma', 'borrow', 'leaner', 'craven', 'deprav', 'freeli', 'salmon', 'deploy', 'imprec', 'saniti', 'receiv', 'breezi', 'sunder', 'fuller', 'lionel', 'expect', 'declar', 'medium', 'polici', 'unsaid', 'deviat', 'colder', 'signal', 'plumag', 'arisen', 'anonym', 'weapon', 'scream', 'husham', 'mishma', 'incumb', 'spindl', 'darwin', 'bottom', 'immort', 'barren', 'pewter', 'filial', 'crackl', 'arteri', 'superl', 'uneasi', 'heaver', 'missiv', 'oddish', 'slogan', 'actuat', 'execut', 'cipher', 'custom', 'fatter', 'actual', 'antill', 'cowhid', 'commit', 'morsel', 'esteem', 'knotti', 'dearth', 'inforc', 'platon', 'impart', 'creagh', 'fillip', 'diagon', 'alight', 'strung', 'viiith', 'dreami', 'remark', 'dampli', 'betray', 'sailer', 'incant', 'excess', 'baptis', 'museum', 'bunger', 'sensit', 'rueful', 'anyhow', 'intrus', 'plaguy', 'tropic', 'ararat', 'priest', 'soften', 'whiter', 'public', 'failur', 'pistol', 'crutch', 'moriah', 'simpli', 'explos', 'squall', 'tatter', 'horrid', '7000l', 'salvat', 'notion', 'strict', 'peltri', 'option', 'hungri', 'select', 'midday', 'sentri', 'vesper', 'effort', 'cordon', 'unbias', 'achiev', 'restor', 'walter', 'unwarp', 'hannib', 'deepen', 'edward', 'murmur', 'cadenc', 'lascar', 'bandag', 'employ', 'ineleg', 'capaci', 'dorsal', 'weaker', 'feeder', 'spoilt', 'unhors', 'latter', 'behold', 'incarn', 'enrich', 'helmet', 'judith', 'strove', 'retail', 'hugest', 'valley', 'longev', 'depart', 'timber', 'geneva', 'unlock', 'pallid', 'lesser', 'prison', 'inflat', 'anchor', 'visibl', 'counti', 'drowsi', 'obstin', 'gothic', 'pamper', 'napkin', 'involv', 'proper', 'requir', 'wisest', 'isaiah', 'freckl', 'circus', 'entrap', 'sketch', 'oxford', 'sitnah', 'nahath', 'outliv', 'splice', 'bodili', 'marlin', 'shabbi', 'social', 'granit', 'potent', 'amount', 'assign', 'shroud', 'boston', 'diadem', 'narrat', 'fetter', 'strang', 'boiler', 'glassi', 'reward', 'crimin', 'delici', 'supper', 'smuggl', 'gander', 'mystif', 'cougar', 'physic', 'entail', 'pastil', 'refuge', 'bilhah', 'hussar', 'unvari', 'interf', 'orison', 'sudden', 'export', 'keeper', 'underw', 'suppli', 'malign', 'infant', 'denial', 'whatev', 'extrem', 'notifi', 'promin', 'thrash', 'ambigu', 'desper', 'nathan', 'naamah', 'tyrant', 'aslant', 'report', 'diamet', 'harder', 'tripod', 'samlah', 'resort', 'cloudi', 'welcom', 'hempen', 'briton', 'implic', 'inflam', 'sparkl', 'disast', 'gallon', 'huzza', 'walker', 'leader', 'belief', 'tenpin', 'frenzi', 'cometh', 'adroit', 'winter', 'lawgiv', 'unclad', 'octavo', 'globul', 'whaler', 'looker', 'terror', 'faceti', 'taurus', 'pacifi', 'knight', 'wrangl', 'coloni', 'ruptur', 'irradi', 'struck', 'clumsi', 'asleep', 'ground', 'trembl', 'afford', 'tempor', 'glutin', 'menial', 'wrinkl', 'corrod', 'impalp', 'underl', 'clench', 'lament', 'pastur', 'adieux', 'presid', 'shobal', 'mincer', 'weazel', 'sinner', 'donkey', 'jackal', 'exclam', 'expand', 'corpus', 'unnear', 'phidia', 'gentil', 'minist', 'moveth', 'savori', 'bengal', 'unpack', 'deacon', 'startl', 'robert', 'lumber', 'thimbl', 'johnni', 'argosi', 'unseal', 'throne', 'sittin', 'whistl', 'endear', 'howdah', 'bidden', 'pluckt', 'edific', 'career', 'eleven', 'unsoci', 'obtain', 'vacanc', 'paunch', 'resent', 'birmah', 'pommel', 'lecher', 'suitor', 'sunken', 'licens', 'ritual', 'fascin', 'togeth', 'recoil', 'violet', 'baptiz', 'shrank', 'enquir', 'manner', 'manass', 'precis', 'cutlet', 'vintag', 'method', 'charit', 'whittl', 'bluish', 'intuit', 'fourth', 'reproa', 'avaric', 'mayhap', 'preval', 'assidu', 'harley', 'redden', 'scrupl', 'hermet', 'plight', 'adrift', 'impetu', 'deligh', 'sensat', 'gardin', 'pedlar', 'gobern', 'gemini', 'flower', 'wrench', 'favour', 'cowley', 'holloa', 'stanza', 'canaan', 'tiller', 'sundri', 'vassal', 'curvet', 'frozen', 'terrif', 'pagoda', 'leakag', 'gloomi', 'worker', 'antoni', 'blanch', 'collis', 'domest', 'abject', 'mohawk', 'silenc', 'hushim', 'banist', 'errant', 'effici', 'shindi', 'careen', 'hereof', 'notabl', 'player', 'unfath', 'exampl', 'toilet', 'advert', 'pardon', 'forsak', 'arnold', 'propos', 'decemb', 'uneven', 'costum', 'keener', 'quebec', 'unreli', 'volley', 'lavend', 'commot', 'clutch', 'drench', 'pester', 'decent', 'silver', 'daggoo', 'articl', 'snuffl', 'fisher', 'egress', 'bullet', 'stupor', 'gesner', 'cannon', 'athlet', 'cosmet', 'irksom', 'twitch', 'tahiti', 'finish', 'submit', 'thrown', 'signif', 'higher', 'waiter', 'brindl', 'valour', 'sayest', 'tremor', 'shinab', 'baboon', 'genera', 'lancer', 'bucket', 'unneed', 'carcas', 'storag', 'novemb', 'ejacul', 'feebli', 'analyt', 'region', 'polish', 'flagon', 'confer', 'slouch', 'blight', 'warfar', 'terrac', 'analog', 'oxygen', 'absurd', 'linear', 'foamin', 'tellin', 'richer', 'unwont', 'poplar', 'embrac', 'enclos', 'wigwam', 'brawni', 'effulg', 'either', 'fornic', 'seaman', 'unseen', 'tester', 'madman', 'screen', 'seduct', 'glossi', 'acquit', 'prefix', 'strife', 'craggi', 'kinder', 'search', 'unhing', 'halloo', 'crotch', 'sprout', 'wriggl', 'jeremi', 'audaci', 'unkind', 'jacket', 'dispel', 'person', 'especi', 'layeth', 'danish', 'hurrah', 'residu', 'abridg', 'allevi', 'unshun', 'heroic', 'enslav', 'halloa', 'repugn', 'intent', 'fathom', 'reuben', 'tribul', 'sacred', 'repent', 'unfail', 'magnet', 'piggin', 'mediat', 'surmis', 'outlet', 'embalm', 'devolv', 'reveng', 'collat', 'helter', 'contin', 'travel', 'unheed', 'lectur', 'copper', 'hidden', 'mighti', 'blackl', 'quaker', 'talbot', 'offenc', 'cannot', 'herein', 'accent', 'unfair', 'socrat', 'attent', 'length', 'confid', 'badest', 'contus', 'taller', 'frigat', 'bonnet', 'mutton', 'frolic', 'mortar', 'speech', 'sichem', 'hostil', 'susten', 'retrac', 'convex', 'gestat', 'renown', 'touchi', 'unfeel', 'romanc', 'disloc', 'window', 'talker', 'impair', 'plenti', 'except', 'cottag', 'junior', 'infirm', 'integr', 'record', 'cylind', 'sophia', 'unduli', 'captur', 'dragon', 'mutual', 'fortun', 'daresn', 'miriam', 'barber', 'invari', 'decoct', 'packet', 'seneca', 'derick', 'mutini', 'wester', 'guilti', 'street', 'attain', 'atroci', 'leisur', 'piquet', 'stipul', 'shrunk', 'feloni', 'reaper', 'batten', 'cooler', 'frivol', 'mayhew', 'breech', 'succor', 'revers', 'eshcol', 'tribut', 'trebli', 'tablet', 'assent', 'unfear', 'unloit', 'forget', 'fixtur', 'pulsat', 'accurs', 'simoon', 'tinker', 'depend', 'enhanc', 'regard', 'approv', 'balena', 'perdit', 'colleg', 'extent', 'vibrat', 'direst', 'herebi', 'barest', 'beater', 'divorc', 'unless', 'pillow', 'friski', 'brevet', 'gallow', 'eldaah', 'famili', 'guinea', 'ashbel', 'tattoo', 'replac', 'potion', 'hudson', 'burden', 'caviti', 'balaen', 'jovial', 'strive', 'deduct', 'wilder', 'eatest', 'mizzen', 'justif', 'outsid', 'reason', 'dramat', 'seclus', 'rascal', 'consol', 'soever', 'attend', 'beriah', 'parent', 'critic', '200th', 'scragg', 'elleri', 'speckl', 'goodbi', 'shiver', 'member', 'hammer', 'machir', 'obtrud', 'almost', 'ingenu', 'morter', 'peniel', 'insist', 'steepl', 'unspot', 'mayest', 'shelah', 'norway', 'overdo', 'muffin', 'achill', 'descri', 'loiter', 'unhoop', 'nautic', 'regina', 'cognis', 'genuin', 'dreamt', 'truest', 'cathol', 'kennel', 'combin', 'subsid', 'algier', 'ambiti', 'textur', 'recurr', 'robust', 'prejud', 'velvet', 'biteth', 'damsel', 'quarri', 'tranqu', 'banish', 'profit', 'ration', 'skewer', 'sprang', 'pliabl', 'invert', 'hinder', 'eyelid', 'sydney', 'bethel', 'scheme', 'marshi', 'pequod', 'fencer', 'stylit', 'outset', 'qualif', 'specif', 'repres', 'euclid', 'curios', 'whenev', 'baroni', 'basket', 'accost', 'direct', 'kidnap', 'center', 'housew', 'sacrif', 'niphon', 'animos', 'import', 'rememb', 'rigger', 'deeper', 'sprain', 'insect', 'peliss', 'honest', 'luggag', 'unviti', 'broken', 'cowper', 'relish', 'afloat', 'repass', 'elijah', 'invent', 'chatti', 'defici', 'vivifi', 'august', 'tyrann', 'believ', 'census', 'attack', 'offend', 'tripli', 'samson', 'threat', 'urchin', 'offspr', 'exceed', 'growth', 'incess', 'woraci', 'chorus', 'habeat', 'legate', 'banker', 'thread', 'penuel', 'monday', 'stitch', 'weight', 'palmer', 'melodi', 'willow', 'labour', 'aggrav', 'hindoo', 'shaveh']
        k=  1
        ['mediocr', 'helpmat', 'pannier', 'poorest', 'generos', 'silveri', 'reparte', 'marriag', 'obstetr', 'apparit', 'midship', 'rabelai', 'gastric', 'judgmat', 'disprov', 'jackson', 'tendril', 'mission', 'bennett', 'adolesc', 'subject', 'triumph', 'indubit', 'highest', 'cashier', 'ghostli', 'demselv', 'marshal', 'healthi', 'skelter', 'encount', 'scraggi', 'slacken', 'cluster', 'lullabi', 'crystal', 'venison', 'oarsmen', 'deepest', 'condemn', 'special', 'iroquoi', 'hatband', 'aground', 'trysail', 'laceped', 'bendigo', 'artisan', 'herring', 'dentist', 'ghastli', 'violenc', 'feebler', 'evermor', 'western', 'shoemak', 'neutral', 'lawless', 'unpaint', 'ledyard', 'stanhil', 'outnumb', 'leapest', 'admonit', 'pageant', 'extinct', 'mislead', 'bermuda', 'commerc', 'ptolemi', 'tumbler', 'bracton', 'exploit', 'unicorn', 'paralys', 'illimit', 'unthink', 'equival', 'distant', 'intreat', 'exhilar', 'ridicul', 'farther', 'traceri', 'unsweet', 'eventid', 'betoken', 'immitig', 'illiber', 'petrifi', 'savanna', 'annuiti', 'concept', 'comport', 'unmatch', 'faileth', 'symmetr', 'appreci', 'whitest', 'bambooz', 'account', 'bastill', 'backbon', 'inscrib', 'dissect', 'jokshan', 'unsourc', 'through', 'tumultu', 'jaundic', 'reflect', 'twelfth', 'morgana', 'iceberg', 'rejoind', 'lonesom', 'underiv', 'immacul', 'feather', 'findeth', 'unhaunt', 'undergo', 'tempest', 'ligatur', 'subtler', 'awkward', 'unmerit', 'babyish', 'shatter', 'encircl', 'undesir', 'persian', 'bullock', 'wayward', 'liberti', 'sentenc', 'festoon', 'indomit', 'bailiff', 'artific', 'nurseri', 'contigu', 'english', 'huskili', 'turmoil', 'meshach', 'scarlet', 'stupifi', 'upstair', 'profess', 'buoyant', 'corlear', 'knowest', 'injunct', 'gregari', 'transom', 'feelest', 'segment', 'selfwil', 'stomach', 'dismiss', 'sunward', 'treasur', 'knoweth', 'broader', 'bequest', 'unyield', 'preserv', 'unfeign', 'sixteen', 'element', 'eccentr', 'antwerp', 'flicker', 'legitim', 'idolatr', 'recreat', 'cetolog', 'straddl', 'whoever', 'ramadan', 'disguis', 'arvadit', 'unfract', 'imprint', 'unscrew', 'chesnut', 'soldier', 'exposur', 'sailmak', 'younger', 'mention', 'tonight', 'anywher', 'typhoon', 'stander', 'kindest', 'opinion', 'disturb', 'submerg', 'suspici', 'succour', 'inconst', 'signifi', 'proverb', 'sleight', 'conceal', 'useless', 'swagger', 'colnett', 'undoubt', 'workman', 'mammoth', 'unsight', 'javelin', 'subsist', 'unstagg', 'swedish', 'needeth', 'fantast', 'palpabl', 'earring', 'request', 'bugbear', 'barbecu', 'kinsmen', 'shutter', 'resembl', 'appoint', 'qualiti', 'canvass', 'librari', 'cervant', 'defunct', 'cruiser', 'sabbath', 'boyhood', 'meanest', 'nothing', 'steerer', 'hemlock', 'agassiz', 'cycloid', 'bewitch', 'raimond', 'solomon', 'wantest', 'aesthet', 'longeth', 'leeward', 'ineffac', 'stretch', 'henshaw', 'antarct', 'servant', 'parsley', 'rehears', 'unremov', 'enamour', 'quadrup', 'embattl', 'callest', 'jetheth', 'blemish', 'recount', 'impious', 'burgess', 'coxcomb', 'paradis', 'mainten', 'karnaim', 'vishnoo', 'tallest', 'heinous', 'haughti', 'ostrich', 'naphish', 'suspens', 'spencer', 'sharpen', 'incumbr', 'meantim', 'smaller', 'indigen', 'garnish', 'nuisanc', 'perplex', 'sartain', 'serious', 'bulwark', 'raiment', 'maachah', 'bejuggl', 'reelman', 'husband', 'hauteur', 'indecis', 'unnatur', 'weather', 'current', 'coopman', 'windsor', 'norfolk', 'explain', 'contend', 'sheleph', 'topsail', 'japheth', 'goodwin', 'scatter', 'abreast', 'sabtech', 'epilogu', 'coleman', 'volcano', 'bolivia', 'walfish', 'satisfi', 'mongrel', 'central', 'ancient', 'sweeter', 'aggriev', 'gudgeon', 'plunder', 'convinc', 'overrun', 'suitabl', 'inboard', 'conceiv', 'general', 'scholar', 'midmost', 'obsequi', 'collaps', 'surveil', 'audienc', 'undress', 'drizzli', 'unhappi', 'doltish', 'perpetu', 'commiss', 'pleasur', 'theoret', 'willing', 'tellest', 'fulller', 'moistur', 'scrambl', 'rigadig', 'curseth', 'marmora', 'whitish', 'resettl', 'incivil', 'succoth', 'succeed', 'savouri', 'nailest', 'eyelash', 'disgrac', 'propuls', 'jettest', 'thorkil', 'cabinet', 'diminut', 'brimmer', 'nineveh', 'firesid', 'harvard', 'somehow', 'couldst', 'proport', 'disobey', 'chapter', 'upright', 'freight', 'perfidi', 'involut', 'counter', 'cruelli', 'sweeper', 'cutlass', 'academi', 'spinoza', 'satieti', 'sensibl', 'ditcher', 'slavish', 'uncordi', 'veriest', 'bulbous', 'guidanc', 'distend', 'redoubl', 'novelti', 'instead', 'monarch', 'unlight', 'nuptial', 'reddish', 'cockpit', 'epitaph', 'thought', 'inflict', 'rhenish', 'creatur', 'prentic', 'fibrous', 'forward', 'pretens', 'foregon', 'rodmond', 'volatil', 'bounder', 'persist', 'disband', 'diaboli', 'fashion', 'banquet', 'slayeth', 'treatis', 'noisier', 'assumpt', 'muezzin', 'minutia', 'unassur', 'habitud', 'vinegar', 'devious', 'meshech', 'striven', 'officio', 'surtout', 'command', 'disgust', 'monsoon', 'engross', 'aladdin', 'luckili', 'seventh', 'himself', 'booksel', 'augment', 'dubious', 'thither', 'genteel', 'charley', 'asenath', 'hampton', 'transit', 'nappish', 'inadequ', 'ziphion', 'herdmen', 'inestim', 'convivi', 'predica', 'porpois', 'whereat', 'wotteth', 'unscath', 'vagrant', 'forcibl', 'ungrasp', 'refrain', 'watcher', 'manikin', 'thinker', 'student', 'clatter', 'kitchen', 'starvat', 'portabl', 'ezekiel', 'opposit', 'cistern', 'lanyard', 'glidest', 'convuls', 'sanguin', 'dention', 'topmost', 'syracus', 'appetit', 'betaken', 'pioneer', 'nourish', 'invalid', 'vowedst', 'spiceri', 'estrang', 'chronic', 'untutor', 'unrival', 'grampus', 'graduat', 'rectori', 'figuera', 'dismast', 'unlimit', 'inhuman', 'pretext', 'ontario', 'shyness', 'breadth', 'chowder', 'dissent', 'barbacu', 'increas', 'unfrequ', 'showest', 'exchang', 'histori', 'languor', 'beneath', 'dolphin', 'cassock', 'keyston', 'manxman', 'lakeman', 'wildest', 'triniti', 'suggest', 'shudder', 'victual', 'jollili', 'peddlin', 'hearken', 'seaport', 'surplus', 'baromet', 'abstemi', 'stiffli', 'percept', 'spasmod', 'bondmen', 'algerin', 'sanctum', 'sibbald', 'eternam', 'ephrath', 'gizzard', 'zemarit', 'piousli', 'stammer', 'ingrati', 'fearest', 'omnipot', 'fresher', 'plantat', 'scrupul', 'athirst', 'mankind', 'absolut', 'articul', 'million', 'pattern', 'forerun', 'newland', 'concert', 'unchang', 'tuileri', 'reclaim', 'provinc', 'further', 'johnson', 'cupbear', 'vicious', 'detract', 'readili', 'protest', 'hatchet', 'rinaldo', 'amittai', 'heathen', 'progeni', 'coincid', 'apostol', 'wharton', 'turkish', 'sinecur', 'corrupt', 'poverti', 'univers', 'torment', 'scissor', 'obeyest', 'tahitan', 'calomel', 'enthron', 'crammer', 'cordial', 'toilett', 'connect', 'remaind', 'torrent', 'cheapli', 'rechurn', 'irrever', 'wearili', 'tidiest', 'niagara', 'elector', 'braveri', 'defianc', 'dissuad', 'untouch', 'galliot', 'compass', 'spiracl', 'mizraim', 'wouldst', 'british', 'negativ', 'obvious', 'egotist', 'springi', 'rebound', 'spontan', 'mezahab', 'clemenc', 'analysi', 'faction', 'imbecil', 'batteri', 'alabama', 'gallant', 'chimney', 'grizzli', 'possess', 'poultri', 'mahomet', 'unconqu', 'obliter', 'demigod', 'warrant', 'tranquo', 'palisad', 'gratifi', 'oversho', 'concoct', 'whiskey', 'breviti', 'bystand', 'shorten', 'postpon', 'enshrin', 'apertur', 'caramba', 'trivial', 'consult', 'ireland', 'distanc', 'protect', 'capstan', 'dungeon', 'grammar', 'enchant', 'undefil', 'retaken', 'hadoram', 'practis', 'disgorg', 'almodad', 'gamesom', 'inexcus', 'caution', 'hardili', 'describ', 'menserv', 'happier', 'assyria', 'quicken', 'abstain', 'griffin', 'malevol', 'whosoev', 'concret', 'tigress', 'pilgrim', 'unhesit', 'abstrus', 'discuss', 'muffled', 'surgeon', 'mandrak', 'inscrut', 'bumpkin', 'deposit', 'secreci', 'keenest', 'complic', 'charnel', 'leopard', 'jahzeel', 'remnant', 'chaotic', 'bespeak', 'bristol', 'permiss', 'graviti', 'whisker', 'popayan', 'brandon', 'dispens', 'girlish', 'mansion', 'classic', 'descent', 'highway', 'lipless', 'sweeten', 'jocular', 'impenit', 'gunpowd', 'unstudi', 'pandect', 'chamber', 'blunder', 'unweari', 'renegad', 'lexicon', 'butcher', 'scorpio', 'supplic', 'journal', 'conclus', 'therein', 'shimron', 'painter', 'bargain', 'zebulun', 'shadowi', 'ottoman', 'instanc', 'reproof', 'entrail', 'portman', 'convers', 'exhaust', 'patienc', 'cylindr', 'remount', 'januari', 'hydrant', 'tillest', 'greater', 'suscept', 'fallaci', 'innumer', 'testili', 'freedom', 'orchard', 'conspir', 'submiss', 'impetus', 'samphir', 'ubiquit', 'african', 'boldest', 'presenc', 'depress', 'largest', 'cassino', 'fisheri', 'attempt', 'gradual', 'prevail', 'hearted', 'proffer', 'colicki', 'russian', 'trowser', 'maccabe', 'burglar', 'control', 'mysteri', 'overlap', 'dryness', 'flatten', 'earlier', 'bladder', 'realiti', 'moonlit', 'leummim', 'scupper', 'cracker', 'sustain', 'concess', 'charger', 'latitud', 'portent', 'simpson', 'merrier', 'expound', 'trellis', 'uncouth', 'saddest', 'webster', 'uncanon', 'tolland', 'citadel', 'overbal', 'royalti', 'sceptic', 'intrins', 'founder', 'heavier', 'impregn', 'gluepot', 'junctur', 'undilut', 'unchart', 'alexand', 'asphalt', 'straggl', 'circumv', 'insipid', 'persuas', 'compact', 'pitcher', 'conveni', 'andiron', 'bentham', 'jimmini', 'impieti', 'societi', 'glorifi', 'checker', 'jahleel', 'harpoon', 'pitiabl', 'workmen', 'hydriot', 'written', 'remembr', 'concubi', 'hereaft', 'dodanim', 'plaster', 'mortifi', 'methink', 'anatomi', 'gateway', 'decorum', 'holborn', 'curvicu', 'waggish', 'unsettl', 'similar', 'bedevil', 'methusa', 'conduct', 'process', 'sharpli', 'fainter', 'narwhal', 'messeng', 'version', 'unavail', 'miracul', 'harbour', 'perceiv', 'perform', 'centaur', 'pendant', 'larceni', 'matsmai', 'nostril', 'fictiti', 'entranc', 'carthag', 'categut', 'incauti', 'modesti', 'revelri', 'thahash', 'sleeper', 'seekest', 'gabriel', 'celesti', 'hoister', 'mealtim', 'fantasi', 'dumbest', 'pretend', 'thinner', 'critter', 'buoyanc', 'soonest', 'conjoin', 'breaker', 'unsubdu', 'uncrack', 'whimsic', 'potenti', 'enigmat', 'cheever', 'america', 'avignon', 'subserv', 'anticip', 'unmitig', 'curricl', 'subsequ', 'boarder', 'dauphin', 'scrabbl', 'statist', 'felicit', 'skysail', 'imparti', 'hackney', 'passant', 'degener', 'consequ', 'wherebi', 'stylish', 'tendenc', 'gilbert', 'economi', 'honiton', 'piteous', 'wardrob', 'nowaday', 'attitud', 'hominum', 'unbound', 'thunder', 'phrensi', 'charter', 'saladin', 'digniti', 'context', 'ellison', 'dispers', 'dawlish', 'lieuten', 'omnivor', 'seventi', 'unsulli', 'worship', 'digress', 'annawon', 'generat', 'pollard', 'spiritu', 'solitud', 'crooked', 'primari', 'unblink', 'zeboiim', 'rampart', 'conclud', 'adulter', 'denunci', 'bethink', 'sideway', 'anguish', 'theolog', 'lantern', 'discipl', 'lacquer', 'chivalr', 'atheism', 'isolato', 'auspici', 'pretenc', 'tension', 'purchas', 'colonel', 'surrend', 'footpad', 'convert', 'omnisci', 'vindict', 'product', 'rostrat', 'exasper', 'puritan', 'dreamer', 'peasant', 'gaseous', 'complet', 'entangl', 'stagger', 'quarter', 'barnacl', 'masteri', 'hisself', 'tremour', 'insepar', 'hostess', 'pursuer', 'unlucki', 'persuad', 'congreg', 'william', 'carriag', 'effectu', 'windpip', 'compris', 'padlock', 'bondman', 'unendur', 'prestig', 'continu', 'overdon', 'ashante', 'lookest', 'azimuth', 'siberia', 'footfal', 'revolut', 'edgewis', 'barouch', 'triangl', 'malacca', 'corlaer', 'dampier', 'tilburi', 'nearest', 'tyerman', 'weighti', 'harmoni', 'potluck', 'sceneri', 'keturah', 'untaint', 'announc', 'plummet', 'albicor', 'catskil', 'comfort', 'perseus', 'magazin', 'norland', 'extract', 'fairest', 'wrought', 'expedit', 'privaci', 'prevent', 'laureat', 'crupper', 'contort', 'sharper', 'magistr', 'abraham', 'thicket', 'kingdom', 'contrit', 'shaster', 'timnath', 'faculti', 'doxolog', 'classif', 'ecstasi', 'eyeless', 'comment', 'overtak', 'elparan', 'canticl', 'unearth', 'stiffen', 'freshen', 'unappar', 'brigger', 'dessert', 'eliphaz', 'clearer', 'valiant', 'blubber', 'havilah', 'bedford', 'jidlaph', 'outlast', 'exagger', 'dissert', 'bedward', 'unwound', 'caprici', 'station', 'unremit', 'medicin', 'mockeri', 'inglori', 'chatter', 'parlour', 'poniard', 'lapland', 'chastis', 'hardest', 'sociabl', 'section', 'catcher', 'emphasi', 'premium', 'definit', 'arsacid', 'easiest', 'invuner', 'tarnish', 'densiti', 'platter', 'against', 'brisson', 'bowsmen', 'discret', 'proclam', 'ricketi', 'plaguey', 'unalloy', 'innkeep', 'without', 'lighter', 'radianc', 'displac', 'resourc', 'unenerv', 'overpow', 'concuss', 'bastion', 'lobtail', 'curtain', 'tweezer', 'consort', 'shallow', 'isthmus', 'arkansa', 'denomin', 'vermont', 'prudent', 'moodili', 'partial', 'mandibl', 'phaedon', 'ellasar', 'smitten', 'between', 'pressur', 'baptizo', 'inherit', 'ascript', 'mustard', 'amplifi', 'vancouv', 'wentest', 'unbegun', 'prodigi', 'princip', 'magdiel', 'conting', 'glacier', 'fiddler', 'provend', 'atheist', 'mystifi', 'juvenil', 'everyth', 'frantic', 'foolish', 'baptism', 'distens', 'surcoat', 'cranium', 'septemb', 'disagre', 'consist', 'confess', 'stringi', 'heroism', 'irrevoc', 'anxieti', 'perturb', 'sixpenc', 'farewel', 'hillock', 'ishmael', 'exorbit', 'talkest', 'impeach', 'dilapid', 'tornado', 'enviabl', 'babylon', 'multipl', 'syllabl', 'chagrin', 'jebusit', 'chilian', 'candour', 'garment', 'abbrevi', 'formosa', 'drunken', 'buffalo', 'deborah', 'monoton', 'foreign', 'perfect', 'protrud', 'disclos', 'melvill', 'condens', 'pizarro', 'sandpap', 'euphrat', 'victori', 'consent', 'patente', 'piercer', 'disport', 'paregor', 'railway', 'surpris', 'correct', 'herbert', 'quicker', 'tuition', 'uninvit', 'charact', 'patient', 'project', 'mangrov', 'irrepar', 'tortois', 'crimson', 'gayhead', 'testifi', 'intoler', 'diametr', 'italian', 'emperor', 'pennant', 'payment', 'conceit', 'antidot', 'philipp', 'heavili', 'shelter', 'pertain', 'solicit', 'thyself', 'untrack', 'aimless', 'skipper', 'destroy', 'uncheer', 'enfeebl', 'discern', 'fritter', 'blamabl', 'scollop', 'commenc', 'balloon', 'swaller', 'warrior', 'unknown', 'slaveri', 'enquiri', 'exhibit', 'sportiv', 'justifi', 'catarrh', 'inveter', 'prouder', 'beseech', 'apparel', 'endless', 'garneri', 'publish', 'travers', 'broider', 'breweri', 'tisburi', 'cellini', 'steward', 'peddler', 'membran', 'pupella', 'overrul', 'heartwo', 'cricket', 'equanim', 'pudding', 'suction', 'bayonet', 'imprimi', 'undecay', 'carcass', 'raphael', 'rowlock', 'turnpik', 'unrustl', 'oppress', 'whereso', 'artless', 'lifetim', 'clamber', 'donavan', 'belfast', 'pomatum', 'eastern', 'jointur', 'practic', 'briefli', 'pildash', 'content', 'souther', 'disastr', 'thereon', 'disdain', 'serpent', 'cushion', 'juggler', 'pyramid', 'hideous', 'blacken', 'mariann', 'handsom', 'compens', 'phantom', 'thicken', 'exclaim', 'channel', 'uncloud', 'varieti', 'lamatin', 'memento', 'protrus', 'flatter', 'convict', 'sympath', 'earnest', 'tedious', 'flutter', 'jingler', 'bespatt', 'rattler', 'countri', 'pursuit', 'midwint', 'conduit', 'prudenc', 'tougher', 'ungener', 'blister', 'dissolv', 'unshorn', 'ailment', 'pedigre', 'languag', 'bandbox', 'messmat', 'benevol', 'prosper', 'sumatra', 'unavoid', 'convent', 'germain', 'delight', 'sorceri', 'unguard', 'selfish', 'masonri', 'evangel', 'commend', 'brawler', 'surplic', 'undecid', 'mansoul', 'overman', 'zoroast', 'requiem', 'address', 'contour', 'deliver', 'express', 'present', 'terribl', 'barrier', 'despair', 'ebullit', 'britain', 'diction', 'illumin', 'elishah', 'decapit', 'suspend', 'success', 'anybodi', 'dantean', 'hastili', 'softest', 'phichol', 'sojourn', 'necklac', 'perenni', 'hangman', 'niggard', 'upbubbl', 'exercis', 'saunter', 'denizen', 'unsleep', 'foretel', 'newspap', 'rhubarb', 'rebekah', 'impress', 'whether', 'rainbow', 'horatii', 'ungradu', 'pointer', 'consign', 'grenadi', 'instant', 'bigotri', 'cherish', 'pugnaci', 'godhead', 'morquan', 'ruinous', 'vatican', 'fervent', 'headway', 'bluster', 'attract', 'moorish', 'placard', 'olassen', 'verdant', 'civilli', 'scougin', 'abandon', 'lighten', 'forebod', 'whereon', 'steamer', 'contriv', 'furnish', 'brother', 'bepatch', 'beverag', 'cruelti', 'assembl', 'violent', 'veteran', 'destini', 'manilla', 'betwixt', 'becharm', 'leather', 'dissoci', 'unpanel', 'magnifi', 'footman', 'inuendo', 'tobacco', 'kedemah', 'mazeppa', 'concern', 'gangway', 'liturgi', 'conform', 'holland', 'fifteen', 'portion', 'shammah', 'therebi', 'partook', 'prophet', 'suspect', 'swollen', 'symptom', 'ballena', 'crucifi', 'cobbler', 'cambric', 'angular', 'plowdon', 'apricot', 'tremend', 'chariot', 'commune', 'kneepan', 'problem', 'drought', 'terrifi', 'diamond', 'billion', 'unpleas', 'seaward', 'settler', 'bowsman', 'bedroom', 'obsolet', 'michael', 'shechem', 'jackass', 'inquiri', 'infinit', 'longest', 'valuabl', 'riphath', 'outwork', 'sitteth', 'riotous', 'oarsman', 'scrutin', 'ecuador', 'unright', 'nonsens', 'shipmat', 'manipul', 'manhood', 'halibut', 'overcom', 'regular', 'profund', 'neither', 'inordin', 'summari', 'whichev', 'preemin', 'draught', 'moulder', 'bethuel', 'unbuckl', 'whitwel', 'topmaul', 'freewil', 'warmest', 'falsifi', 'infiltr', 'mistifi', 'unstabl', 'douceur', 'tighten', 'jolliti', 'incoher', 'journey', 'doctrin', 'thicker', 'passeng', 'surpass', 'wealthi', 'unalter', 'geometr', 'discont', 'squalli', 'walketh', 'potteri', 'bilious', 'fiercer', 'headach', 'merrili', 'glimmer', 'dislodg', 'chariti', 'overcam', 'swarthi', 'seedtim', 'cullest', 'discard', 'brahmin', 'passion', 'frobish', 'concurr', 'pictori', 'holiest', 'militia', 'sheweth', 'trumpet', 'engulph', 'steepli', 'simplic', 'rheumat', 'mermaid', 'flexion', 'immingl', 'tarquin', 'collect', 'fratern', 'unappal', 'gaffman', 'missent', 'imageri', 'concili', 'fastidi', 'midsumm', 'contact', 'rephaim', 'harpstr', 'allegor', 'bernard', 'lehabim', 'certain', 'blanket', 'voucher', 'feminam', 'boister', 'deliber', 'haggard', 'terraqu', 'jeopard', 'farrago', 'molucca', 'slappin', 'biggest', 'conquer', 'relianc', 'lustili', 'shinbon', 'panther', 'amongst', 'holiday', 'japanes', 'reliabl', 'wherein', 'anteced', 'oversig', 'qualifi', 'unquiet', 'despond', 'burnish', 'illiter', 'haarlem', 'giddili', 'creativ', 'beliest', 'uninjur', 'whither', 'ungraci', 'hickori', 'applaud', 'bannist', 'traitor', 'maritim', 'dyspept', 'congeal', 'rounder', 'overtur', 'willain', 'trodden', 'spheric', 'hapless', 'markest', 'temptat', 'carpent', 'hogarth', 'enkindl', 'lookout', 'upbraid', 'respond', 'forfeit', 'slender', 'quarrel', 'quietud', 'shallop', 'skylark', 'dunkirk', 'congeni', 'binnacl', 'critiqu', 'honesti', 'flannel', 'selfsam', 'furlong', 'massacr', 'accumul', 'shillem', 'receipt', 'oneself', 'auction', 'insular', 'swallow', 'antelop', 'blossom', 'butteri', 'dearest', 'freshet', 'speaker', 'greener', 'fattest', 'auditor', 'tuesday', 'ugliest', 'infidel', 'illustr', 'almanac', 'glazier', 'feminin', 'uncount', 'enliven', 'findest', 'flinger', 'undaunt', 'tragedi', 'hammock', 'unsound', 'travail', 'centuri', 'interim', 'slipper', 'kinross', 'horrifi', 'buckler', 'plethor', 'impolit', 'biscuit', 'goodwil', 'visitor', 'monopol', 'forborn', 'sprinkl', 'epaulet', 'benefit', 'abimael', 'constel', 'scallop', 'trapper', 'whereof', 'engraft', 'popular', 'kingham', 'tableau', 'furious', 'aliment', 'glisten', 'osseous', 'confirm', 'develop', 'villain', 'noblest', 'compani', 'inspect', 'majesti', 'annihil', 'enderbi', 'languid', 'barbari', 'anomali', 'factori', 'vapouri', 'glitter', 'respect', 'captain', 'aptitud', 'likewis', 'retreat', 'courier', 'foresaw', 'vacuiti', 'bolster', 'martial', 'crucibl', 'wrapper', 'camphor', 'uniform', 'crozett', 'doorway', 'refresh', 'chiefli', 'essenti', 'elliott', 'punctur', 'angrili', 'filigre', 'precaut', 'afflict', 'kremlin', 'halyard', 'consecr', 'windrow', 'struggl', 'woollen', 'acceler', 'porthol', 'carrion', 'floweri', 'repetit', 'indecor', 'grapnel', 'sulphur', 'depreci', 'boatmen', 'coalesc', 'forbear', 'mummeri', 'cathedr', 'ephraim', 'burgher', 'smackin', 'grecian', 'authent', 'chicken', 'reticul', 'evanesc', 'circuit', 'various', 'egyptia', 'monster', 'anxious', 'rayther', 'complac', 'prolong', 'codfish', 'predict', 'resound', 'runaway', 'affront', 'iceland', 'jealous', 'decreas', 'unmanag', 'technic', 'scratch', 'unassum', 'terrier', 'builder', 'holburn', 'purport', 'suffici', 'inhabit', 'illinoi', 'swimmer', 'gershon', 'sunshin', 'uncivil', 'fernand', 'staunch', 'cholera', 'ensconc', 'flexibl', 'cranial', 'support', 'gentler', 'particl', 'squilge', 'environ', 'incrust', 'probabl', 'spouter', 'imperil', 'gettest', 'lazarus', 'possibl', 'generic', 'counsel', 'amphibi', 'redoubt', 'brought', 'sunbeam', 'nervous', 'bravado', 'strello', 'scorbut', 'postman', 'england', 'bravest', 'cutleri', 'shrivel', 'outward', 'knocker', 'wettest', 'holdest', 'viceroy', 'harvest', 'immoder', 'wiseish', 'recross', 'twopenc', 'credibl', 'penalti', 'snuffer', 'goddess', 'unspeak', 'neglect', 'resolut', 'whisper', 'unflatt', 'renounc', 'counten', 'slander', 'showeri', 'laudabl', 'dismemb', 'villani', 'wolfish', 'hornpip', 'propens', 'happili', 'whereto', 'galleri', 'contest', 'plaudit', 'throttl', 'display', 'envious', 'vignett', 'daugher', 'envelop', 'assault', 'thomson', 'descend', 'horizon', 'fullest', 'recruit', 'smother', 'cetacea', 'horribl', 'pirouet', 'relievo', 'overlay', 'caravan', 'firmest', 'partner', 'canakin', 'zealand', 'curious', 'foundat', 'obstacl', 'meeteth', 'thereof', 'nosegay', 'distort', 'zealous', 'alreadi', 'facilit', 'entreat', 'contain', 'ballast', 'radiant', 'dignifi', 'slumber', 'corsair', 'patriot', 'equinox', 'entrust', 'pervers', 'spanish', 'sometim', 'repress', 'bandana', 'altitud', 'duskier', 'proceed', 'forlorn', 'nichola', 'pharaoh', 'ochotsh', 'undivid', 'rollick', 'bloodsh', 'respons', 'richard', 'hastier', 'leicest', 'sherbet', 'bourbon']
        k=  2
        ['', 'litig', '105', 'bedad', 'sconc', 'ward', 'vivid', 'from', 'slack', 'begun', 'unrel', 'cough', 'wick', 'help', 'rein', 'quail', 'bluff', 'lave', 'crop', 'buckl', 'rous', 'omit', 'arbor', 'aris', 'mapl', 'glebe', 'bring', 'seer', 'avast', 'ere', 'agenc', 'tash', 'torn', 'oper', 'least', 'chock', 'allay', 'scarc', 'cook', 'alvah', '93', 'river', 'guid', 'brand', 'gild', 'hate', 'sofa', 'tame', 'eliza', 'quoin', 'psalm', 'frock', 'facil', 'hon', 'clove', 'nurs', 'curl', 'mamr', 'ob', 'hvalt', 'duck', 'feat', 'indig', 'shall', 'ingin', 'blast', 'tingl', 'czar', 'paler', 'mall', 'wharv', 'os', 'jew', 'dedic', 'manli', 'ape', 'score', 'well', 'clew', 'fain', '1842', 'repar', 'earth', 'fool', '1779', 'brace', 'wash', 'vat', 'lose', 'lass', 'bag', 'sloop', 'allur', 'eye', 'salem', 'slave', 'steep', 'surg', 'numer', 'tomb', 'chao', 'sayst', '200', 'becom', 'push', 'rich', 'vine', 'egi', 'thine', 'brush', 'dread', 'screw', 'vi', '1825', 'desir', 'reu', 'prick', 'text', 'amid', 'peddl', '1788', 'bann', 'tack', 'endow', 'sir', 'stare', 'abus', 'giggl', 'lucif', 'jalap', 'hose', 'unwan', 'repin', 'l1', 'unman', 'graze', 'bay', 'hobah', 'hain', 'asiat', 'seek', 'cloth', 'peopl', 'ge', 'mean', 'feud', 'bark', 'porus', 'itali', 'jame', 'extra', 'pla', 'impot', 'sleek', 'ii', 'forti', 'liken', 'leo', 'winc', 'light', 'gale', 'taunt', 'soil', 'by', 'dandi', 'clerk', 'hie', 'rat', 'archi', '2', 'sharp', 'wavi', 'imit', 'truer', 'misl', 'jobab', '1778', 'pledg', 'toil', 'hey', 'learn', 'vero', 'wise', 'cub', 'to', 'crab', '5th', 'still', 'poor', 'slice', 'vertu', 'weep', 'bulli', 'gerar', 'fawn', '1726', 'beor', 'floe', 'peter', 'knack', 'rid', 'poss', 'phil', 'emot', 'duell', 'vein', 'spoon', 'truth', 'ashor', 'frost', 'stab', 'parti', 'volum', 'am', 'who', 'coffe', 'prow', 'stout', 'naval', 'see', 'tola', 'penal', 'earl', 'seeva', 'gain', 'east', 'din', 'flank', 'fie', '124', '63', 'rippl', 'dealt', 'fishi', 'derid', 'boozi', 'surf', '62', '440', 'focus', 'hunk', 'knot', 'inlay', 'hosea', 'sleep', 'bust', 'quiet', 'delta', 'chuse', 'john', 'inaud', 'exact', 'state', '130', 'shadi', 'auger', '44', 'saw', 'smut', 'brain', 'loan', 'georg', 'ari', 'droll', 'mere', 'spare', 'ablut', 'buri', 'coher', 'clink', 'belik', 'law', 'azor', '133', 'lien', 'host', 'wing', 'reduc', 'mare', 'fir', 'hobbl', 'crisi', 'toll', 'laps', 'beef', 'gnaw', 'proa', 'fieri', 'calam', 'slink', 'went', 'foami', 'ahoy', 'uno', 'jar', 'noder', 'sourc', 'cloud', 'rabbl', '1695', 'clap', 'hap', 'bide', 'clave', 'scud', 'sulki', 'oaken', '1668', 'strap', 'rigid', 'ceti', 'begon', 'shone', 'steal', 'buck', 'braid', 'bone', 'all', 'hero', 'salut', 'herd', 'nt', 'wade', 'vehem', 'behr', 'grant', 'inert', 'gruff', 'buoy', 'jebb', 'caput', 'spake', 'gat', 'alacr', 'larg', 'furi', 'foibl', 'blade', 'loung', 'earn', 'about', 'nestl', 'plaid', 'nail', 'ounc', 'giddi', 'jacob', 'citi', 'wet', 'del', 'scald', '58', 'saith', 'log', 'timor', 'pick', 'orbit', 'rig', 'rome', 'cap', 'lone', 'undec', 'nobl', 'flue', 'their', 'ever', 'brook', 'statu', 'baser', 'anti', 'curdl', 'bout', 'sprat', 'night', 'tu', 'none', 'tread', 'huff', 'ghast', 'marbl', 'trail', 'nuee', 'hurri', 'chanc', 'chees', 'jingl', 'clank', 'aroma', '17', 'snowi', 'hemp', 'bank', 'ninni', 'deifi', 'like', 'liabl', 'axe', 'hecla', 'stud', 'ee', 'hadst', 'stair', 'scamp', 'slate', 'oakum', 'tract', 'ineff', 'yawn', 'linen', 'gout', 'stuf', 'truli', 'label', 'noth', 'pat', 'arrog', 'diddl', 'dilat', 'roost', 'flout', 'tap', 'skin', 'sew', 'knoll', 'forth', 'galli', 'worm', 'alleg', 'abeam', 'sidon', 'plait', 'begat', 'slid', '28', 'mad', '29', 'unit', 'plato', 'bress', 'tax', 'tower', 'afric', 'bibl', 'fanci', 'oahu', 'wagon', 'emigr', 'were', '72', 'sylph', 'grave', 'kin', 'shine', 'five', 'ki', '89', 'hors', 'beat', 'tent', 'add', 'vent', 'obe', 'baffl', 'dusk', 'seem', 'thar', 'broth', 'lin', 'pout', 'seir', 'clear', 'wire', 'lobbi', 'weakl', 'awok', 'spade', 'alvan', 'gaham', 'apex', 'class', 'tool', 'zag', 'play', 'santa', 'groan', 'onli', 'kill', 'moos', 'grape', 'belay', '1793', 'hue', 'citat', '1750', 'pow', 'haul', 'oft', 'liter', 'intim', 'twig', 'zidon', 'haran', 'hish', 'envi', 'silv', 'wap', 'lack', 'hapli', 'exult', 'mat', 'fierc', 'luck', 'bliss', 'su', 'lain', 'exot', 'sneak', 'mss', 'jaw', 'davi', 'aga', 'yanke', 'inkl', 'smote', 'prynn', 'boy', 'swiss', 'meagr', 'drift', 'stain', 'isl', 'excit', 'hurtl', 'ripe', 'lest', 'drawl', '6', 'cubit', 'this', '45', 'abov', 'unmov', 'yew', 'allus', 'avenu', 'heart', 'palav', 'ship', 'venic', 'bond', 'dart', 'annus', 'tim', 'hoari', 'broad', 'layn', 'glee', 'repos', 'rage', 'estat', 'run', 'brine', 'perri', 'brass', 'wine', '101', 'sedat', '121', 'hear', 'secur', 'penni', 'flew', 'hast', 'hori', 'touch', 'decor', 'shock', 'stove', 'satan', 'chap', 'baulk', 'goal', 'petul', 'guest', 'crane', 'snug', 'spoke', 'wee', 'ain', 'teaz', 'fou', 'thoma', 'day', 'gore', 'ridg', 'v', 'pier', 'lie', 'an', 'lusti', 'occas', 'gaff', 'flyin', 'rebut', 'chart', 'anew', 'ire', '37', 'uniqu', 'salah', 'medit', 'style', 'feed', 'agon', 'hust', 'unvex', 'fate', '30', 's', 'brim', 'one', 'rave', 'jeush', 'snarl', 'pluck', 'flap', 'troil', 'skirt', 'burst', 'dumb', 'ice', 'can', 'champ', '117', 'entri', 'dash', '46', 'rapid', 'due', 'budg', 'etch', '120', 'twisk', 'list', 'marl', 'have', 'hour', 'x', 'elam', 'foggi', 'glue', 'knob', 'beari', 'hall', 'brig', 'idol', 'wafer', 'bipe', 'order', 'hagar', 'm', 'cash', 'palmi', 'si', 'hang', 'korah', 'anno', 'jostl', 'dwarf', 'tore', 'gulp', 'refer', 'n', 'decis', 'decre', 'rent', 'deput', 'whoel', 'my', 'dumbl', 'veri', 'hobb', 'awak', 'mace', 'west', 'ill', '106', 'neck', 'star', 'ruin', 'beaux', 'trend', 'wheat', 'copi', 'ride', 'judah', 'are', 'spite', 'insid', 'adjac', 'hundr', 'greek', 'excav', 'fume', 'into', 'advic', 'ing', 'rip', 'impal', 'slime', 'zohar', 'feder', 'elud', 'paper', 'divid', 'crook', 'ann', 'card', '96', 'pious', 'sever', 'bye', 'lunat', 'sammi', 'sort', 'oili', 'tan', 'train', 'dip', 'gnarl', 'bleak', 'how', 'girl', 'kelpi', 'shi', 'bere', 'abash', 'semin', 'cruet', 'opul', 'in', 'alien', 'wood', 'quilt', 'essex', 'till', 'flaki', 'sette', 'laugh', 'hitti', 'mari', 'flung', 'itch', 'blue', 'harm', 'boobi', 'lump', 'poki', 'nill', 'flip', 'cheek', 'amiss', 'size', 'soot', 'peop', 'peal', 'ban', 'apiec', 'leap', 'hater', 'libra', '1st', 'revel', 'india', 'home', 'vile', 'farc', 'union', 'mute', 'basso', 'hunch', 'dull', 'isol', 'caw', 'act', 'stack', 'pond', 'daunt', 'pipe', 'led', 'fe', 'lent', 'fifti', 'annex', '81', 'ajah', 'ohad', 'hill', 'goggl', 'polit', 'harsh', 'here', 'moon', 'hob', 'keto', 'palsi', 'sere', 'hoop', 'flea', 'sell', 'paean', 'lustr', 'tint', 'cone', 'divis', 'kant', 'taper', 'trump', 'vow', 'rapt', 'non', 'mown', 'hood', 'jesu', 'engin', 'swarm', 'judea', 'hardi', 'affin', 'love', 'mutin', 'mam', 'bosom', 'relic', 'fault', 'plead', 'engag', 'cane', '109', 'uz', 'tar', 'cove', 'pounc', 'rhode', 'move', 'seest', 'oh', '1819', 'diver', 'brisk', 'sarah', 'eagl', 'moder', 'trait', 'ork', 'idler', 'whelp', 'doesn', 'ceil', 'lug', 'bushi', 'spe', '74', 'thick', 'lurid', 'claw', 'said', 'whip', 'major', 'prize', 'ooz', 'bend', 'unmix', 'hoe', 'tinkl', 'coke', 'pea', 'dote', 'tissu', 'gnash', 'goest', 'worri', 'refil', 'nick', 'brown', 'mist', 'mout', 'mate', 'lai', 'alert', 'paul', 'port', 'jona', 'hold', 'bounc', 'infin', 'spell', 'onyx', 'pivot', 'amain', '49', 'dr', 'conic', 'leaki', 'when', 'acr', 'fuss', 'orgi', 'marri', 'foie', 'maker', 'smack', 'ne', 'demur', 'aft', 'jumbl', 'wheez', 'parvo', 'brute', 'maci', 'dim', 'torso', 'dumpl', 'rames', 'wild', 'give', 'elig', 'pedro', 'nomin', 'sooth', 'assyr', 'emolu', 'croni', 'chuck', 'juic', 'lotus', 'cat', 'send', 'tip', 'dood', 'trust', 'mogul', 'castl', 'shown', 'zoar', 'deris', 'puff', 'fare', 'net', 'path', 'l100', 'aslop', 'vast', '150', 'jac', 'thee', 'fact', 'mous', 'wale', 'loon', '111', 'elaps', 'prime', 'indit', 'cool', 'meddl', 'cubic', 'eider', 'bode', 'choos', 'debat', 'tuft', 'leech', '61', 'chose', 'just', 'kink', 'undet', 'speci', 'dolli', 'today', 'owe', 'gas', 'lint', 'lade', 'accad', 'dere', 'child', 'yaw', 'amber', 'penem', 'sub', 'optic', 'awe', 'gray', 'activ', 'badg', 'alew', 'rose', 'fluke', 'two', 'four', '104', 'break', 'leash', 'ebal', 'gor', 'charl', 'told', 'short', 'adhes', 'entic', 'beech', '90', '84', 'calah', 'pose', 'wield', 'flirt', 'fine', 'sob', 'visag', 'alloy', 'cover', 'slili', 'main', 'esek', 'silli', 'lip', 'avers', 'invad', 'trim', 'cramp', '131', 'shad', 'den', 'thing', 'chese', 'vide', 'race', 'width', 'marin', 'gash', 'ri', 'livid', 'smoke', 'upsid', 'case', 'valor', 'dot', 'refus', 'pink', '1850', 'hasti', 'untim', 'europ', 'vale', 'oust', 'thump', 'sweat', 'posit', 'elb', 'fig', 'weri', 'gr', 'world', 'adder', 'shaft', 'kid', 'parri', 'nasti', 'macey', 'supin', 'navig', 'capt', 'sedul', 'becam', 'paley', 'groov', 'demon', 'big', 'bleed', 'mile', 'yeast', 'find', 'canni', 'gold', 'vacat', 'amal', 'weld', 'ahaz', 'audac', 'sleev', 'jerah', 'do', 'wrote', 'dug', 'yet', 'count', 'drawn', 'monit', 'sed', 'bess', 'knell', 'eleph', 'latch', 'unsay', 'imper', 'mood', 'darkn', 'bera', 'stub', 'utter', 'radic', 'of', 'glim', 'no', 'prove', 'talk', 'zar', 'merri', 'close', 'gaza', 'hee', 'draft', 'haunt', 'evinc', 'cheap', 'hay', 'hurl', 'tempt', 'f', 'maid', 'lamp', 'dure', 'voic', 'acrid', 'zuzim', 'flock', 'fell', 'wil', '87', 'deleg', 'han', 'toad', 'bier', 'angri', 'way', 'fang', '114', 'booth', 'yarn', 'block', 'sage', 'may', 'coven', 'razor', 'bid', 'gorg', 'paid', 'roli', 'ey', 'sworn', 'trial', 'opera', 'often', 'vener', 'wit', 'knew', 'gown', 'scold', 'groom', 'rise', 'forg', 'file', 'admit', 'olden', 'guis', 'abaft', 'alow', 'niger', 'rule', 'noon', 'limb', 'wasp', 'ach', 'pegu', 'waist', 'benam', 'toddi', 'mose', 'theme', 'save', 'cynic', 'addit', 'hack', 'unwit', 'battl', 'dutch', 'viol', 'spurn', 'coal', 'dwell', 'turn', 'recit', 'germ', 'wrap', 'rivet', 'elev', 'onset', 'fungi', 'whizz', 'dodg', 'dri', 'dowri', 'knead', 'now', 'deck', 'tough', 'snort', 'quid', 'bade', 'navel', 'hou', 'lock', 'erron', 'yoke', 'simpl', '4', 'canst', 'arpen', 'tong', 'pud', 'sulk', 'shot', '73', 'huge', 'spray', 'tog', 'laden', 'honey', 'breez', 'sooti', 'mode', 'p', 'spool', 'reev', 'site', 'misti', 'knive', 'erupt', 'apart', 'spine', 'feroc', 'black', 'wept', 'rack', 'reg', 'anah', 'hirah', 'fuel', 'space', 'duke', 'huz', 'fife', 'oliv', 'pent', 'elv', 'shun', 'sheet', 'burn', 'denot', 'wor', 'bolt', 'dinah', 'comi', 'deem', 'queen', 'diet', 'corps', 'spend', 'blang', 'popt', 'belat', 'slain', 'glanc', 'moab', 'phase', 'sheav', 'harri', 'paus', 'berg', 'valu', 'mungo', 'doom', 'more', 'merg', 'dong', 'desk', 'ham', '25', 'spi', 'jezer', 'sam', 'daven', 'depos', 'civil', 'take', 'near', 'baron', 'or', 'waiv', 'agent', 'glove', 'fagot', 'ensu', 'celer', 'quart', 'lung', 'catch', 'bland', 'rope', 'afoul', 'insur', 'slyli', 'subt', 'erech', 'sli', 'old', 'sheba', 'alon', 'oth', 'asia', 'mild', 'rider', 'pork', 'true', 'whiff', 'estim', 'durst', 'ate', 'puppi', '40', 'bush', 'swing', 'make', 'chill', 'labor', 'doubl', 'fata', 'se', 'spike', 'leaf', 'awri', 'singl', 'gait', 'reput', 'll', 'rigor', 'cloak', 'mob', 'unnot', 'crust', 'wager', 'aveng', '79', 'brag', 'malay', 'clark', 'god', 'gihon', 'maw', 'quest', 'wharf', 'soon', 'hole', 'stave', 'susan', 'beard', '108', 'jabal', 'reel', 'aton', 'map', 'ambit', 'turtl', 'grin', 'tran', 'uner', 'tidi', 'plump', 'spawn', 'arid', 'real', 'fresh', 'ur', 'harem', 'if', 'abel', 'gette', 'haven', 'earli', 'ludim', 'lowli', 'scorn', 'blew', 'iron', 'siev', 'queer', 'dam', 'born', 'glad', 'timna', 'off', 'cana', 'jag', 'jelli', 'wa', 'greas', 'dear', 'rude', 'feint', 'pore', 'blot', 'comer', 'swore', '38', 'templ', 'won', 'brat', 'cross', 'jungl', 'whelm', 'sting', 'rosh', 'azur', 'ile', 'scalp', 'ralli', 'dedan', 'item', 'stern', 'spire', 'clock', 'waken', 'freez', 'impud', 'chode', 'prey', 'pill', 'saul', '134', 'lift', 'teman', 'jonah', 'fed', 'as', 'mirth', 'flit', 'guido', 'crew', 'lay', 'toder', 'stash', '60', 'goa', 'biddi', 'air', 'thenc', 'spos', 'scent', 'ezbon', 'spain', 'zig', 'equat', 'witti', 'york', '11', 'globe', 'betim', 'flo', 'unti', 'amaz', 'cajol', 'issu', 'seed', 'gilt', 'hotel', 'meet', 'bread', 'fat', 'motiv', 'wear', 'shade', 'jub', 'keel', 'didst', 'metal', 'cosi', 'gig', 'ewer', 'sent', 'strut', 'toe', 'plate', 'aloud', '14', 'ford', 'puls', 'cave', 'doe', 'event', 'on', 'readi', 'pickl', 'raw', 'crow', 'solus', 'must', 'lofti', 'foam', 'pri', 'weig', 'fling', 'doat', 'chase', 'unset', 'w', 'peke', 'past', 'bet', 'skip', 'upris', 'will', 'kenit', 'throw', 'outer', 'sad', 'spici', 'ruf', 'fire', 'back', 'egg', 'sawn', 'bound', 'hung', 'caddi', 'mede', 'bowi', 'loam', 'drugg', 'churn', 'bruit', 'seal', 'toast', 'fuse', 'jiff', 'read', 'aid', 'h', 'harto', 'gloom', 'domin', 'warn', 'storm', 'pike', 'gag', 'hurt', 'coron', 'pupil', 'comic', 'sauci', 'poni', 'cedar', 'toppl', 'befel', 'andes', 'fabul', 'akin', 'whoso', 'mead', 'motto', 'palac', 'sang', 'throe', 'boast', 'week', 'quit', 'belli', 'ruffl', 'anon', 'yes', 'terah', 'mantl', '23', 'smeer', 'swamp', 'shalt', 'busk', 'aspir', 'creak', 'matr', 'gouti', 'ger', '1820', 'flesh', '88', 'load', 'quick', 'yearn', 'fun', 'altar', 'angel', 'punch', 'water', 'lower', 'intox', 'chick', 'perus', 'chief', 'loud', 'daze', 'ant', 'noos', 'kedar', 'heeva', 'we', 'jare', 'lili', 'unwed', 'infix', 'deep', 'piti', 'knit', 'minut', 'milki', 'pre', 'cigar', 'handl', 'aglow', 'parm', 'tawni', 'step', 'tarri', 'humph', 'joist', '68', 'spear', 'fleet', 'lit', 'stead', 'aisl', 'mell', 'orb', 'flume', 'bulb', 'dwe', 'babe', 'reef', 'l', 'rustl', 'saco', 'owest', 'kit', 'lieth', 'savor', 'expel', 'inde', 'book', 'hist', 'betak', 'lane', 'cram', 'oar', '48', 'riv', 'aleak', 'quaff', 'shuni', 'drank', 'versa', 'odd', 'crown', 'shewn', 'mecca', 'sac', 'hitch', 'xvi', 'swept', 'tutor', 'dewi', 'noway', 'isaac', 'peril', 'luci', 'hush', 'most', 'sing', 'seiz', 'heth', 'hesit', 'heath', 'rock', 'jesus', 'room', 'pour', 'wan', 'nois', 'swoon', 'pant', 'legal', 'ghent', 'tebah', 'leav', 'print', 'onam', 'emin', 'hu', 'morev', 'brunt', 'beg', 'dial', 'studi', 'press', 'wake', 'virgo', 'owner', 'snap', 'cope', 'upper', 'ohio', 'sail', 'march', 'clout', 'clan', 'avert', 'bustl', 'raft', '35', 'anim', 'crape', 'flash', 'chalk', 'low', 'grub', 'rate', 'bak', 'match', 'argo', 'quito', 'spent', 'also', 'o', 'mani', 'snow', 'thole', 'juba', 'crim', 'pig', 'irad', 'mossi', 'bar', 'fag', 'burnt', 'cart', 'hous', 'hazo', 'appar', 'kneel', '113', 'lava', 'daft', 'weav', 'poli', 'sledg', 'popul', 'mound', 'dock', 'sog', 'filer', 'alo', 'froth', 'whew', 'wal', 'tow', 'serah', 'encas', 'blown', 'gall', 'zebul', 'bluer', 'veda', 'unlik', 'rabid', 'resum', 'mine', 'usual', 'insul', 'cuba', 'lord', 'bridg', 'rust', 'untri', 'wrapt', 'quog', 'patch', 'sound', 'rash', 'mumbl', 'coral', 'shod', 'herb', '1846', 'avoc', 'drum', 'druri', 'inn', 'ledg', 'solac', 'trudg', 'eleg', 'parch', 'bind', 'mufti', 'therm', 'ye', 'ebon', '129', 'pen', 'disk', 'balm', 'defin', 'coll', 'frown', 'mid', 'clime', '94', 'broil', 'crumb', 'cavil', 'cain', 'hunt', 'irish', '10', 'fore', 'toper', '115', 'pile', 'rue', 'jewel', 'walk', 'remun', 'rotat', 'nose', 'where', 'gami', 'curd', 'glu', 'injur', 'stori', 'curs', 'arion', 'ombay', 'tost', '85', 'self', 'brood', 'delic', 'cheat', 'tooth', 'began', 'hake', 'tube', 'what', 'vagu', 'yore', 'be', 'shank', 'funni', 'brawn', 'pebbl', 'rake', 'ribbi', 'same', 'six', 'tema', 'corn', 'volit', 'finit', '102', 'gull', 'damn', 'stilt', '5', '59', 'rag', 'khan', 'type', 'tranc', 'orion', 'worn', 'mock', 'dust', 'nahor', 'say', 'inmat', '91', 'antic', 'manx', 'bred', 'box', '95', 'jaffa', 'iscah', 'docil', 'humbl', 'poem', 'spung', 'vacil', 'babi', 'hark', 'cradl', 'embay', 'fur', 'sick', 'clung', 'duti', 'tone', 'pisc', 'name', 'repel', 'atom', 'anak', 'tubal', 'flask', 'regul', 'awar', 'riddl', 'tast', 'tune', 'oli', 'greet', 'vocat', 'epher', 'st', 'drunk', 'theft', 'cote', 'beast', 'head', 'came', 'stun', 'maxim', 'slat', 'goe', 'boobl', 'bough', 'sixth', 'dung', 'virul', 'duff', 'bench', 'down', 'maul', 'adult', 'halv', 'needl', 'coerc', 'ball', 'thaw', '4th', 'ard', 'teat', 'rebel', 'guilt', 'nat', 'tiger', 'foal', 'rev', 'daili', 'pump', 'dine', 'birth', 'eden', 'ail', 'jesti', 'lasso', 'thorn', 'belov', 'admir', 'gam', 'tray', 'fle', 'zarah', 'asid', 'adam', 'agre', 'windi', 'beer', 'eve', 'negat', 'cattl', '1833', 'evok', 'delus', 'dab', 'would', 'ibid', 'tail', 'vane', 'wrest', 'stopt', 'stem', '31st', 'hire', 'hypo', 'sit', 'club', 'haint', 'deed', 'hare', 'slow', 'obeis', 'mess', 'gomer', 'triun', 'blind', 'baden', 'na', 'him', 'tic', 'leach', 'dozen', 'vers', 'loop', 'nate', 'turf', 'patri', 'hull', 'petti', 'dog', 'tiara', 'wage', 'gang', 'ephah', 'trick', 'bias', 'clump', 'gurri', 'trebl', 'cheer', 'safe', 'unlov', '1652', 'claim', 'band', 'bela', 'choic', 'so', 'peru', 'grey', 'ript', 'thou', 'g', 'sperm', 'figur', '77', 'darkl', 'clay', 'mizen', 'glade', 'front', 'apeak', 'riot', 'nanci', 'squid', 'unami', 'sikok', 'dad', 'gayer', 'recur', 'cog', 'leant', '12', 'allot', 'zerah', 'smoki', 'dig', 'feege', 'them', 'sign', 'bare', 'adorn', 'goug', 'prebl', '1729', 'befor', 'milit', '132', 'wed', 'joosi', 'cours', 'medal', 'candi', 'broom', 'i', 'expir', 'nudg', 'oak', 'sway', 'tacit', 'bled', 'clapt', 'live', 'tulip', 'twill', 'nabob', 'felic', 'small', 'alli', 'best', 'enrol', 'mosqu', 'might', 'face', '100', 'ring', 'floor', 'hop', 'cadiz', 'tith', 'scant', 'abort', 'plung', 'jane', 'poke', 'toler', 'could', 'loos', 'delud', 'blith', 'paint', 'defer', 'feign', 'grief', 'huron', 'start', 'pull', 'medan', 'chasm', 'climb', 'tierc', 'trade', 'rove', '80', 'en', 'joki', 'puzzl', '7', 'pau', 'pint', 'ungod', 'fifth', 'nigh', 'clad', 'rul', 'scrap', 'alfr', '1821', 'hoot', 'unent', 'th', 'bulk', 'axi', 'con', 'hem', 'pole', 'use', 'wait', 'crash', 'rail', 'breed', 'chink', 'soap', 'mint', 'high', 'beli', 'calai', 'howl', 'hazel', 'titl', 'flint', 'stand', 'age', 'click', 'iniqu', 'duli', 'hing', 'trap', 'flaw', 'frame', 'gros', 'frail', 'gush', '000', 'peel', 'isra', 'isui', 'crime', '36', 'stuff', 'swam', 'teach', 'board', 'fair', 'comb', 'advoc', 'kick', 'sterl', 'abram', 'uzal', 'tyre', 'eh', 'vari', 'pacif', 'sauc', 'amus', 'bandi', 'drop', 'me', 'quill', 'rattl', 'mow', 'echo', 'boug', 'scarf', 'dagon', 'newer', 'check', 'tudor', 'yea', 'blush', 'stump', 'jago', 'pass', 'hamul', 'inion', 'warp', 'pouch', 'befal', 'level', 'horit', 'after', 'meali', 'twin', 'menac', 'ladi', 'polic', 'alcov', 'cost', 'wool', 'piec', 'stew', 'rusti', 'crept', 'magic', 'asa', 'found', 'pair', 'lurch', 'wolv', 'ammon', 'lbs', 'ermin', 'tit', 'gum', 'token', 'look', 'writ', 'arrow', 'crazi', 'juli', 'congo', 'oscil', 'hath', 'dazzl', 'spit', 'smelt', 'eclat', 'vault', 'pearl', 'slept', 'ibi', 'bung', 'bob', 'minc', 'folk', 'pave', 'oat', 'cell', 'split', 'canal', 'valv', 'shur', 'bodic', 'invis', 'amuck', 'unlac', 'afor', 'thro', 'hail', 'swim', 'helm', 'growl', 'aunt', 'store', 'wife', 'probe', 'glide', 'bre', 'foul', 'fail', 'ober', 'anvil', 'scare', 'muscl', 'spat', 'rain', 'fege', 'steer', 'creat', 'song', 'liest', 'holi', 'imbib', 'young', 'gross', 'grand', 'dalli', 'wind', 'art', 'bitt', 'eav', 'canon', '52', 'sole', 'meant', 'stock', '107', 'shave', 'fiddl', 'allow', 'unabl', 'fickl', 'caulk', 'taint', 'knave', 'hadad', 'giant', 'truce', 'awl', 'seth', 'smite', 'sware', 'attar', 'staid', 'pain', 'stalk', 'shut', 'grove', 'music', 'wink', 'ugli', 'debt', 'cent', 'len', 'feet', 'ferri', 'gird', 'tight', 'barb', 'few', 'poser', 'groin', 'juri', 'rot', 'unev', 'guard', 'exig', 'manag', 'lath', 'hook', 'ma', 'lick', 'musk', 'rumbl', 'prowl', 'boil', 'divin', 'amend', 'erect', 'cluni', 'older', 'villa', 'asund', 'unab', 'snaki', 'gun', 'adapt', 'whet', '1851', 'beadl', 'hith', 'mati', 'abhor', 'murki', 'moot', '20', 'chew', 'idiot', 'untag', 'danc', 'jig', 'muski', 'stake', 'steam', 'cock', 'ale', 'royal', 'lar', 'abl', 'heav', 'impel', 'plain', 'shed', 'arrah', 'insan', 'cush', 'zeal', 'sum', 'repli', 'flee', 'ir', 'tick', 'abr', 'inher', 'herod', 'widen', 'plume', 'ani', 'oxen', 'vorac', 'mab', 'rear', 'al', 'flow', 'veer', 'awhil', 'subdu', 'swift', 'darbi', 'kindr', 'faith', 'obal', 'faeri', 'rode', '13', 'elop', 'local', 'lee', 'sud', 'poet', 'stool', 'mamma', '51', 'equal', 'grudg', 'novel', 'kenaz', 'buy', 'paran', 'babel', 'bass', 'web', 'reub', 'hivit', 'sneez', 'aorta', 'go', 'piqu', 'sin', 'tree', 'palat', 'cut', 'yell', 'vicar', 'treat', 'idiom', 'appel', 'dan', 'tie', 'reced', 'king', 'neat', 'keg', 'serva', 'escap', 'met', 'solar', 'weedi', 'prune', 'ubiqu', 'er', 'soul', 'pilau', 'aloft', 'sate', 'with', 'clam', 'price', 'wore', 'mourn', 'swart', 'wilt', 'witt', 'lo', 'seven', 'cabl', 'miss', 'plum', 'jure', 'hint', 'shrub', 'tipto', 'defil', 'fitz', 'bore', 'feel', 'ophir', 'assum', 'enact', 'axl', 'ting', 'rex', 'sal', 'besid', 'circl', 'speed', 'pyre', 'dint', 'lacer', 'skil', 'lasha', 'fret', 'hor', 'boggi', 'even', '116', 'refin', 'bath', 'stoic', 'next', 'rumin', 'ding', 'rival', 'bruis', 'adopt', 'gard', 'green', 'moodi', 'proud', 'flare', 'lap', '69', 'ghost', 'craze', 'purr', '65', 'logan', 'franc', 'elm', 'stow', 'onc', 'inhal', 'dirti', 'nibbl', 'manor', 'robe', 'boski', 'obedi', 'incap', 'van', 'sling', 'heard', 'orlop', 'gad', 'tend', '15', 'choke', 'beal', 'swell', 'threw', 'lamb', 'bed', 'last', '1772', 'veloc', 'ruddi', 'pray', '135', 'laban', 'colt', 'slunk', 'omin', 'panel', 'flog', 'everi', 'relax', 'ween', 'busi', 'li', 'hit', 'extol', 'eyest', 'tumbl', 'other', 'crowd', 'girth', '1671', 'eight', 'toss', 'kiss', 'ruler', 'fan', 'her', 'irasc', 'pang', 'sapl', 'charm', 'luz', 'firma', 'darl', 'sun', '126', 'rob', 'heron', 'renew', 'sheep', 'blend', 'heigh', 'mrs', 'coil', 'drill', 'fill', 'jail', 'clean', 'job', 'unaid', 'dea', 'fibr', 'wo', 'loo', 'sheer', 'spot', 'heat', 'fuzz', 'serv', 'women', 'is', 'loath', 'mene', 'fear', 'pitch', 'duet', 'tea', 'rap', 'doze', 'knee', 'rosi', 'oa', 'time', 'gentl', '43', 'accus', 'corki', 'upon', 'gripe', 'em', 'solo', 'place', 'imag', '550', 'wari', 'plea', 'whirl', 'muzzl', 'delin', 'perth', 'wide', 'sold', 'sodom', 'metr', '75', 'shell', 'debel', 'sigh', 'tier', 'sabbe', 'inact', 'thi', 'cant', '110', 'slide', 'centr', 'armor', 'pusi', 'vital', 'troop', 'spill', 'clot', 'glori', 'tile', 'invoc', 'fluid', 'howev', 'pimpl', '97', 'salam', 'rural', 'juan', 'waxen', 'expos', 'rung', 'exalt', 'gasp', 'curat', 'chais', '890', 'grown', 'els', 'heap', 'cocoa', 'wolf', 'truck', 'bird', 'dost', 'dark', 'whenc', 'lief', 'amput', 'hai', 'fog', 'among', 'sonor', 'stall', 'moreh', 'grate', 'dusti', 'crier', 'array', 'defi', 'both', 'held', 'java', 'wert', 'chang', 'chime', 'hallo', 'avow', 'imput', 'knock', 'eloqu', 'negro', '1828', 'rank', 'pod', 'famin', 'call', 'tart', 'shel', '53', 'crush', 'durer', 'edar', 'tide', 'stink', 'educ', 'pan', 'poker', 'undon', 'greec', 'opium', 'drink', 'cur', '1775', 'starv', 'pois', 'orang', 'pop', 'cord', 'ezer', 'incur', 'posi', 'bodi', 'flail', 'sight', 'ignit', 'aerat', 'prate', '123', 'tell', 'badn', 'sugar', 'slam', 'blaze', 'jack', 'redol', 'herr', 'twain', 'apron', 'pieti', 'sarai', 'veng', 'jubil', 'mou', 'berri', 'frog', 'aner', 'mummi', 'stigg', 'broke', 'lodg', 'field', 'carmi', 'grisl', 'giver', 'cholo', 'fatal', 'homag', 'u', 'fin', 'griev', 'idea', '9', 'grap', 'coax', 'lank', 'then', 'undu', 'rib', 'arch', 'which', '70', 'bait', 'hussi', 'roger', 'sw', 'wont', 'area', 'haggi', 'refug', 'spice', 'abod', 'patro', 'dole', 'vicin', '55', 'mar', 'snooz', 'heed', 'salin', 'aback', 'princ', 'diaz', 'whose', 'evolv', 'lima', 'ebb', 'tenth', 'bevi', 'deni', 'need', 'gape', 'agit', 'stiff', 'carri', 'ident', 'veil', 'fowl', 'abrah', 'wight', 'shan', 'stop', 'owen', 'magna', 'fled', 'akan', 'te', 'jest', '2000', 'util', 'walw', 'bulg', 'smell', 'enoch', '8', 'whoop', 'drag', 'at', 'shout', 'tak', 'talon', 'gave', 'south', 'lavat', 'empir', 'pampa', 'pool', 'got', 'nape', 'abid', 'yield', 'sabl', '127', 'mass', 'show', 'ex', 'endur', 'unmar', 'faint', 'cleet', 'offic', 'fai', 'thumb', 'lt', 'arodi', 'ado', 'cabin', 'fairi', 'waxi', 'great', 'clo', 'cru', 'funer', 'gazer', 'per', 'enter', 'incid', 'boa', 'code', 'capac', 'ay', 'lingo', 'three', 'durat', 'base', 'tickl', 'hid', 'folio', 'pound', 'rio', 'brack', 'money', 'prose', 'humil', 'myrrh', 'void', 'sweet', 'noun', 'beset', 'moni', 'sap', 'rad', 'bite', 'vail', 'japh', 'deliv', 'right', 'acerb', 'along', 'mama', 'nice', 'chili', 'spin', 'son', 'flake', 'slew', 'pail', 'slung', 'open', 'appl', 'demi', 'arkit', 'mayb', 'adio', 'pert', 'sneer', 'crack', 'lad', 'squat', 'blank', 'wat', 'whit', 'hat', 'sank', 'tri', 'sword', 'shirt', 'lanc', 'half', 'yard', 'occur', 'craft', 'core', 'femal', 'genus', 'rifl', 'left', 'feast', 'page', 'beget', 'staff', 'new', 'skulk', 'less', 'grow', 'j', 'japan', 'judg', 'solec', 'hoard', 'cure', 'male', 'men', 'balli', 'sunni', 'brew', 'numb', 'bowel', 'berth', 'befog', 'pest', 'toy', 'solid', 'trunk', 'stay', 'nest', 'rest', 'littl', '180', 'shift', 'fring', 'aught', 'usag', 'drew', 'dead', 'espi', 'jamin', 'aint', 'limit', 'watch', 'jen', 'wreck', 'spoil', 'jot', 'pin', 'adah', 'tidal', 'tramp', 'hval', 'flush', 'incit', 'brave', 'prais', 'abund', 'free', 'rush', 'sure', 'juggl', 'chess', 'jubal', 'gloss', 'tilt', 'hartz', 'aw', 'rebuk', 'a', 'white', 'creep', 'road', 'senor', 'mould', 'chip', 'urn', 'scyth', 'smith', 'bell', 'impur', 'umbil', 'spile', 'hah', 'grog', 'peer', 'ador', 'texa', 'err', 'mind', 'beak', 'far', 'execr', 'firm', 'soggi', 'bee', 'war', 'kn', 'hoki', 'swear', 'hump', 'willi', 'hard', 'kept', 'straw', 'bloat', 'joe', 'uncov', 'atad', 'blent', 'semi', 'padan', 'kine', 'froid', 'reviv', 'guess', 'musti', 'bent', 'under', 'dilut', '83', 'woman', 'pope', 'bundl', 'adown', 'jump', 'tens', 'clang', 'pate', 'ordin', 'sake', 'mason', 'know', 'seri', 'fidel', 'fli', 'bunk', 'ideal', 'scoot', 'pure', 'lure', 'stubb', 'row', 'wall', 'nig', 'yale', 'elong', 'smile', 'gaze', 'blanc', 'caus', 'curio', 'loyal', 'oder', 'your', 'slab', 'dough', 'heavi', 'stabl', 'vigor', 'leer', 'cupid', 'camel', 'happi', 'chi', 'verd', 'fewer', 'unwil', 'get', 'ahead', 'reuel', 'feje', 'sod', 'plane', 'weal', 'aver', 'sens', 'shame', 'dare', 'veal', '92', 'teak', 'had', 'nake', 'inter', 'resen', 'trot', 'rowel', 'sur', 'ago', 'unus', 'built', 'fish', 'rod', 'dower', 'easi', 'gleig', 'minor', 'wax', 'bridl', 'uncl', 'orat', 'care', 'tunic', 'basi', 'inner', 'tape', 'sagac', 'regal', 'aros', 'sweep', 'infal', 'plumb', 'calv', 'infer', 'quak', 'gift', '86', 'basin', '41', 'overs', 'long', 'clasp', 'humor', 'sea', 'furl', 'north', 'alik', 'shrug', 'slant', 'shore', 'notch', 'scar', 'kettl', 'salli', 'hove', 'arriv', 'dump', 'bald', 'zepho', 'havoc', 'pooh', 'elat', 'lith', 'arbah', 'indi', 'mr', 'cite', '19', 'moan', 'mix', 'isa', 'line', 'fight', 'mower', 'fern', 'shoe', 'allud', 'finic', 'note', 'shirr', 'deft', 'glaze', 'loom', 'scrub', 'fals', 'mite', 'sacr', 'elect', 'twist', 'sock', 'ducat', 'twos', 'asham', 'hone', 'sire', 'adher', 'nowis', 'fit', 'jupit', 'woodi', 'bulki', 'post', 'bunch', 'bill', 'phiz', 'girdl', 'suckl', 'recov', 'urban', 'laid', 'side', 'she', 'bubbl', 'ulcer', 'strip', 'mizz', '118', 'flame', 'mangl', 'teter', 'gulli', 'nativ', 'hemam', 'kindl', 'niec', 'whole', 'lash', 'tom', 'elk', 'sink', 'quig', 'took', 'lunar', 'cork', 'tut', 'edom', 'snore', 'purti', 'rover', 'glen', 'goeth', 'sore', 'serug', 'joker', 'degre', 'emul', 'retak', 'usher', 'indic', 'china', '66', 'raze', 'crete', 'enjoy', 'form', 'texel', 'mule', 'deiti', 'retir', 'out', 'swap', 'reli', 'pulpi', 'carv', '39', 'man', 'tusk', 'lead', 'cargo', 'don', 'accid', 'hatch', 'invas', 'shaul', 'tribe', 'suck', 'seat', 'agil', 'ev', 'cold', 'idl', 'pagan', 'ink', 'raal', 'rend', 'mark', 'husk', 'ark', 'fema', 'pay', 'birch', 'sip', 'moss', 'fork', 'paw', 'heir', 'flood', 'he', 'bronz', 'date', 'dane', 'polar', 'e', 'satin', 'wound', 'error', 'peac', 'advis', 'boot', 'evad', 'slim', 'aliv', 'avith', 'hen', 'exert', 'auto', 'fix', 'set', 'luff', 'bear', 'dairi', 'fleec', 'roll', 'tyro', 'door', 'wonst', 'wail', 'squir', 'hoo', 'rebek', 'c', 'hover', 'drip', 'infam', 'saxon', 'snake', 'begin', 'beach', 'risk', 'first', 'weed', 'minus', 'there', 'sill', 'witch', 'fanni', 'shill', 'boon', 'recal', 'drown', 'gras', 'grunt', 'dilig', 'bless', 'nap', 'rambl', 'bit', 'year', 'link', 'coach', 'ters', 'eel', 'joint', 'curb', 'sist', 'sleet', 'hairi', 'vitus', 'wish', 'felt', 'tekel', 'hul', 'clog', 'bison', 'enrag', '82', 'ampl', 'virtu', '1690', 'sorri', 'harp', 'thiev', 'unaw', 'lid', 'think', 'hair', 'fenc', 'avail', 'pris', 'tish', 'dut', 'shook', 'ravin', 'deer', 'mast', 'bride', 'ash', 'our', 'innat', 'grade', 'grind', 'lover', 'woof', 'sich', 'ear', 'fuego', 'end', 'chast', 'sat', 'bold', 'that', 'impun', 'wring', 'sober', 'betti', 'beam', '31', 'doer', 'seen', 'pip', 'mt', 'doubt', 'plank', 'root', 'haze', 'scene', '50', 'opin', 'been', 'sup', 'jam', 'doest', 'ankl', 'made', 'fall', 'marsh', 'cruis', 'gap', 'imbed', 'whang', 'mutat', 'tardi', 'sunk', 'plagu', 'joke', 'swede', 'loui', 'soak', 'coat', 'har', 'paddl', '57', 'burk', 'blood', 'tackl', 'envoy', 't', 'defac', 'visit', 'sale', 'alp', 'fanat', 'joy', 'swung', 'town', 'deal', 'wrung', 'hell', 'pinon', 'heal', 'resid', 'thigh', 'sex', 'pli', 'goin', 'chafe', 'deter', 'dawdl', 'barqu', 'jet', 'tira', 'ulloa', 'pleas', 'bribe', 'le', 'chace', 'babbl', 'pelvi', 'curv', 'pine', 'eri', 'lace', 'some', 'tuck', 'xerx', 'pale', 'ocean', 'penit', 'looke', 'trace', 'couch', 'henri', 'rever', 'seawe', 'stamp', 'taken', 'syren', 'delay', 'sport', 'bashe', 'ahab', 'organ', 'write', 'thief', 'roast', 'not', 'canva', 'drove', 'bottl', 'jenni', '47', 'mend', 'pleat', 'duski', 'vivac', 'fogo', 'spur', '21st', 'appli', 'annal', 'hot', 'dwelt', '800', 'fetid', 'chum', 'tawn', 'it', 'rocki', '15th', 'gaili', 'hilar', 'ran', 'elder', 'unusu', 'ass', 'womb', 'bull', 'zo', 'unfit', 'verif', 'fob', 'sh', 'serf', 'roomi', 'ivori', 'ea', 'marg', 'hazi', 'risen', 'onan', 'abil', 'bitch', 'stig', 'wand', 'catt', 'angl', 'caper', 'view', '42', 'rind', 'scott', 'selv', 'dome', 'scout', 'salad', 'crisp', 'ka', 'arc', 'sown', 'word', 'octob', 'good', 'mauri', 'belub', 'voi', 'skill', 'tin', 'wiser', 'final', 'below', 'bunt', 'rum', 'ehi', 'stood', 'emir', 'maze', 'spee', 'capit', '1840', 'trope', 'freed', 'affix', 'cri', 'saddl', 'ether', 'weak', 'verac', 'voyag', 'petit', 'senat', 'mole', 'incom', 'hip', 'etc', 'effac', '1839', '3', 'entir', 'madam', 'yon', 'bale', 'foot', '1836', 'scoop', 'milk', 'folli', 'perch', 'nobli', 'app', 'game', 'hum', 'zay', 'chin', 'drug', 'damag', 'coars', 'tee', 'habit', 'was', 'phut', 'til', 'modif', 'zone', 'ratif', 'excel', 'ferul', 'illus', 'anyon', 'over', 'asher', 'devic', 'throb', '21', 'pinch', 'those', 'nun', 'whom', 'oppos', 'life', 'bou', 'astir', 'exhal', 'shem', 'reach', 'jeer', 'fox', 'terra', 'cinqu', 'build', 'immut', 'sou', 'foe', 'iv', 'let', 'keep', 'accru', 'von', '16th', 'feebl', 'initi', 'shark', 'sash', 'edg', 'lard', 'skiff', 'pear', 'lawn', 'hog', 'nor', 'innoc', 'squar', 'dirt', 'invit', 'wipe', 'shape', 'want', 'ophit', 'peg', 'hil', 'fast', 'phara', 'sixti', 'thew', 'brink', 'lama', 'silk', '16', 'evid', 'pot', 'sidel', 'tamar', 'swath', 'serri', 'eager', 'june', 'cage', 'peep', 'hag', 'tru', 'youth', 'crawl', 'fumbl', 'tub', 'fist', 'nors', 'siam', 'calf', 'mesha', 'panic', 'quot', 'es', 'elon', 'leak', 'scale', 'esau', 'bash', 'torch', 'liber', 'hymn', 'doeth', 'gin', 'cano', 'oblig', 'slay', 'pomp', 'loss', 'apt', 'butt', 'such', 'sky', 'they', 'inequ', 'cape', 'immov', 'hoist', 'mask', '56', 'sane', 'gobbl', 'bo', 'rape', '1811', 'flour', 'moral', 'dens', 'proof', 'tete', 'endu', 'encor', 'smirk', 'icili', 'own', 'attun', 'grain', 'coin', 'ergo', 'slipt', 'deaf', 'peck', 'stoop', 'brow', 'handi', 'you', 'swain', 'remov', 'shoal', 'boom', 'rood', '1807', 'agoni', 'ensur', 'solv', 'goat', 'frank', 'lan', 'stark', 'aim', 'barn', 'blunt', 'fame', 'barr', 'rim', 'exet', 'mong', 'mail', 'unnam', 'gate', 'reign', 'mash', 'edit', 'satir', 'finni', 'warbl', 'sunda', 'tho', 'stone', 'seduc', 'whim', 'ultim', 'malic', '33', 'eddi', 'clash', 'bewar', 'fort', 'shake', 'dover', 'moth', 'oasi', 'wean', 'jib', 'index', 'monk', 'park', 'nile', 'fri', 'froze', 'henc', 'cup', 'speak', 'layer', 'jud', 'leagu', 'but', 'gaunt', 'eeli', 'and', '103', 'vomit', 'sinew', 'sow', 'pack', 'tigri', 'plug', 'mane', 'woven', 'hope', 'car', 'damp', 'slit', 'aloof', 'sha', 'thin', 'nimbl', 'io', 'hav', 'sour', 'forc', 'mohr', 'gera', 'human', 'sinit', 'eau', 'tug', 'notic', 'enemi', 'droop', 'junip', 'fac', 'alpin', 'crag', 'ton', 'top', '144', 'sect', 'wrong', 'brit', 'dous', 'scan', 'sung', 'utton', 'stray', 'jamb', 'loaf', 'assur', 'clip', 'fulli', 'wrath', 'navi', 'slap', 'lud', 'rare', 'turk', 'spout', 'dire', 'tepid', 'draw', 'dun', 'roof', 'dey', 'muffl', 'agu', 'hawk', 'chat', 'mill', 'coupl', 'emet', 'seba', 'cling', 'much', 'cozen', 'ad', 'trio', 'latin', 'simil', 'group', '99', 'noisi', 'mammi', 'nook', 'rug', 'adequ', '26', 'pilot', 'lucki', 'oil', 'inabl', 'lime', 'bigot', 'charg', 'pall', 'grace', 'wiv', 'come', '24', 'vex', 'tonic', 'soft', 'worth', 'equip', 'wive', 'never', 'done', 'skull', 'hyena', 'cun', 'armi', 'loin', 'bang', 'grew', 'grasp', 'eat', 'jerk', 'waft', 'cleav', 'glow', 'mud', 'sand', 'lye', 'part', 'nut', 'tongu', 'alb', 'dumah', 'adieu', 'vial', 'friar', 'natio', 'snuf', 'late', 'tag', 'cob', 'brief', 'sinc', 'stir', 'felin', 'ste', 'eber', 'unrig', 'morn', 'steel', 'vexat', 'stifl', 'hilt', 'pott', 'yojo', 'vice', 'mappl', 'merit', 'tr', 'cruiz', 'exil', 'evil', 'his', 'the', 'arm', 'reap', 'grab', 'newli', 'ennui', 'wag', 'savag', 'sa', 'airth', 'track', 'drain', 'guni', 'pit', 'gatam', 'salt', 'lobe', 'vigil', 'via', 'fond', 'pursu', 'rubi', 'mats', 'bloom', 'natur', 'plata', 'iii', 'lieu', 'balmi', 'fever', 'hwal', 'tangl', 'waif', 'kor', 'usurp', 'sylla', 'dant', 'didn', 'fath', 'halt', 'talus', 'emerg', 'levi', 'spar', 'dream', 'synod', 'slash', 'news', 'edict', 'carey', 'immin', 'elus', 'fo', 'rang', 'twine', 'sous', 'swum', 'miser', 'nip', 'grope', 'piano', 'stage', 'alter', 'rais', 'coast', 'dat', 'peleg', 'maim', 'spose', 'snuff', 'twas', 'bow', 'chair', 'baker', 'candl', '3d', 'rhyme', 'seeth', 'behav', 'fruit', 'insol', '400', 'comet', 'devil', 'joppa', 'court', 'vain', 'while', 'clamp', 'wedg', 'meat', 'crave', 'worst', 'crish', 'hoar', 'sle', 'shove', 'tabl', 'weigh', 'beau', 'peak', 'chest', 'decay', '64', 'model', 'pawn', 'writh', 'co', 'farm', 'teeth', 'again', 'weari', 'pelt', 'heel', 'nettl', 'rescu', 'pie', 'gase', 'dale', 'each', 'sled', 'urg', '1492', 'gay', 'ladl', 'glass', 'grim', 'fro', 'steed', 'cleat', 'prong', 'tink', 'argu', 'anger', 'anyth', 'lot', 'leg', 'seam', 'essay', 'drive', 'magog', 'fade', 'grego', 'muddl', 'float', 'vum', 'inch', 'mop', 'mum', 'april', 'alm', 'total', 'swoop', 'dove', 'quaf', 'roar', 'twel', 'wane', 'cake', 'anker', 'bake', 'midst', 'abe', 'pirat', 'ala', 'plini', 'finer', 'trip', 'sack', '500', 'farth', 'pratt', 'fray', 'sto', 'glare', 'verit', 'wave', 'omar', 'gleam', 'enorm', 'brick', 'slip', 'cruel', 'dent', 'pierc', 'crest', 'offer', 'ego', 'koo', 'fetch', 'fasc', 'belt', 'task', 'drab', 'ray', 'death', 'twice', 'work', 'whale', 'grit', 'cream', '1776', 'cast', 'join', 'third', 'duel', 'ten', 'arter', 'topic', 'hide', 'ditto', 'cuss', 'known', 'kith', 'bur', 'ripen', 'barg', 'award', 'rout', 'mazi', 'dame', 'attic', 'scour', 'ram', 'elbow', 'warm', 'bawl', 'hears', 'ha', 'rub', 'palm', 'did', 'imp', 'tire', 'nonc', 'clue', 'gem', 'soar', 'noah', 'lake', 'lurk', 'beeri', 'sickl', 'pison', 'porch', 'prior', 'mug', 'sped', 'enabl', 'until', 'cibil', 'swerv', 'leper', 'pace', 'lull', 'dabol', 'lion', 'annum', 'sheaf', 'skim', 'bask', 'shew', 'wid', 'bead', 'horn', 'boat', 'eno', 'blame', 'da', 'la', 'stuck', 'settl', 'appal', 'jove', 'stapl', 'undul', '2nd', 'tat', 'tenor', 'et', 'lever', 'abat', 'bat', 'fold', 'key', 'l150', 'put', 'pox', 'milch', 'wreak', 'de', 'has', 'honor', 'ignor', 'dar', 'unwel', 'elast', 'given', 'blur', 'empti', 'd', 'up', 'afoam', 'colic', 'buz', 'wors', 'pell', 'tale', 'lirra', 'drama', 'accur', 'ixion', 'bowl', 'anoth', 'gloat', 'stick', 'twelv', 'hug', 'prop', 'gulf', 'beno', 'ox', 'um', 'hoof', 'fuddl', 'cring', 'purg', 'ian', 'thame', 'gabl', '54', 'rogu', '98', 'bump', 'watt', '125', 'dupe', 'tini', 'eas', 'ephra', 'wast', 'smart', 'meal', 'rough', 'cow', 'cliff', 'span', 'muff', 'share', 'jone', 'abas', 'fiend', 'calm', 'woe', 'later', 'matur', 'ho', 'oath', 'freer', 'poop', 'goos', 'liv', 'gees', 'dis', 'ewe', 'ross', 'mingl', 'massa', 'cuff', 'nod', 'elah', 'eulog', 'subtl', 'prone', 'ajar', 'titan', 'purs', 'ceas', 'puf', 'ought', 'ici', 'whist', 'chop', 'amen', 'ren', 'alarm', 'claus', 'mouth', '18', 'die', 'middl', 'dawn', 'trifl', 'red', 'term', 'roman', 'husba', 'belah', 'bud', 'decid', 'kee', 'ah', 'fa', 'opal', 'jolli', 'ask', 'suit', 'javan', 'flat', '22', 'trod', 'tis', 'plan', 'cark', 'knife', 'bad', 'madai', 'memor', 'test', 'drat', 'food', 'wi', 'dive', 'creed', 'unsaf', 'dew', 'gill', 'lend', 'cobbl', 'saint', 'admah', 'bacon', 'indol', 'unrol', 'esq', 'round', 'await', 'jav', 'impos', '27', 'point', 'rumpl', 'these', 'hanov', 'scowl', 'power', 'land', 'liver', 'kiln', 'squaw', 'depth', '119', 'bomb', 'jilt', 'miner', 'shuah', 'etern', 'egypt', 'away', 'gre', 'ou', 'imped', 'dangl', 'anus', '76', 'shoot', 'dick', 'islet', 'talli', 'nay', 'full', 'icicl', 'floc', 'shelv', 'spasm', 'omen', 'tall', 'too', 'tun', 'fetor', 'for', 'oozi', 'parse', 'iram', 'cleft', 'spark', 'bower', 'wheel', 'brows', 'deriv', 'boni', 'eliez', 'liar', '78', 'aram', 'parad', 'gurgl', 'gro', 'alway', 'mote', 'afir', 'desol', 'wot', 'vest', 'shear', 'raven', 'heber', 'pride', 'shop', 'valis', '32', 'mutil', 'aye', 'lost', 'oval', '34', '1', 'slope', 'irrat', 'lege', 'fabl', 'fugit', 'wad', 'mi', 'lotan', 'dropt', 'snare', 'hand', 'shred', 'dish', 'doth', 'pinni', 'nine', 'stole', 'dress', 'grass', 'nati', 'lazi', 'swash', 'godbi', 'thus', 'raini', 'chare', 'amour', 'jetur', '67', 'merci', 'invok', 'eaten', 'blow', 'cask', 'delug', 'nerv', 'thank', 'cuttl', 'afar', 'ampli', 'mayst', 'unto', 'naked', 'obey', 'cetus', 'keen', 'ditch', 'irrit', 'jiffi', 'vase', '1791', 'hadar', 'woo', 'wrist', 'goney', 'mobi', 'limp', 'rumor', '71', 'hamor', 'hiss', 'futur', '112', 'anglo', 'lean', 'bungl', 'alley', 'baton', 'odor', 'impli', 'eboni', 'hind', 'juici', 'exist', 'whi', 'us', 'unequ', 'shelf', 'kind', 'coy', 'separ', 'sag', 'mount', 'plant', 'widow', 'flag', 'relat', 'goad', 'jog', 'vote', 'remot', 'junk', 'bitin', 'pari', 'moor', 'cato', 'quay', 'almon', 'crick', 'fee', 'acut', 'steak', 'devot', 'aran', 'cod', 'timid', 'stung', 'mottl', 'purpl', '128', 'gone', 'foil', '122', 'induc', 'avoid', 'leah', 'melt', 'pad', 'amor', 'seren', 'tear', 'wroth', 'grip', 'chain', 'month']


        Args:
            No input arguments

        Returns:
            Nothing.
            The function is printing in console 3 clusters for words
        '''
    txt1 = preprocessing8(text1)
    txt2 = preprocessing8(text2)
    txt3 = preprocessing8(text3)
    txts = txt1.union(txt2)
    txts = txts.union(txt3)
    txts = list(txts)
    words = np.array([
        [
            distance.edit_distance(w, i)
            for i in txts
        ]
        for w in txts
    ], dtype=np.float64)
    cent, _ = kmeans(words, k_or_guess=3)
    word_cluster = np.argmin([
        [spatial.distance.euclidean(wv, cv) for cv in cent] for wv in words
    ], 1)
    for k in range(cent.shape[0]):
        print('k= ', k)
        print([word for i, word in enumerate(txts) if word_cluster[i] == k])


kmean_compare_K()





