"""
first: copy or extract datasets directory(for us, corpus) on same directory as this python scirpt file
The first job is to bring in everything we need from scikit-learn:---------
"""

import os
import sys
import collections
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan

import time

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.8, 's' : 80, 'linewidths':0}
# 


dataSetDir2 = os.path.join(os.getcwd(), "dataset")

def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    print(labels.shape)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
    plt.show()
    

all_data = datasets.load_files(dataSetDir2, 
    description=None, load_content=True, encoding='utf-8', shuffle=False)
#--------------------------------------------------------------


"""
This is a function that will count the number of times each word in the
dataset occurs and project that count into a vector(TF-IDF vector). 
"""
count_vectorizer = TfidfVectorizer(stop_words='english')

X = count_vectorizer.fit_transform(raw_documents=all_data.data)

svd = TruncatedSVD(n_components=200)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

data = lsa.fit_transform(X)
n_clusters = 14
plot_clusters(data, cluster.KMeans, (), {'n_clusters':n_clusters})

plot_clusters(data, cluster.AffinityPropagation, (), {'preference':-5.0, 'damping':0.95})

plot_clusters(data, cluster.MeanShift, (0.175,), {'cluster_all':False})

plot_clusters(data, cluster.SpectralClustering, (), {'n_clusters':n_clusters})

plot_clusters(data, cluster.AgglomerativeClustering, (), {'n_clusters':n_clusters, 'linkage':'ward'})

plot_clusters(data, cluster.DBSCAN, (), {'eps':1.1, 'min_samples':2})

plot_clusters(data, hdbscan.HDBSCAN, (), {'min_cluster_size':5})