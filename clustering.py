"""
first: copy or extract datasets directory(for us, corpus) on same directory as this python scirpt file
The first job is to bring in everything we need from scikit-learn:---------
"""

import os
import sys
import collections
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import dump_svmlight_file
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.8, 's' : 80, 'linewidths':0}

k_clusters = 10

def kmeans_clustering(datasetDir, preprocessing):
    #  load our data-----------------------------------------
    all_data = datasets.load_files(datasetDir, 
        description=None, load_content=True, encoding='utf-8', shuffle=False)
    #--------------------------------------------------------------
    labels = all_data.target


    """
    This is a function that will count the number of times each word in the
    dataset occurs and project that count into a vector(TF-IDF vector). 
    """
    count_vectorizer = TfidfVectorizer()

    X = count_vectorizer.fit_transform(raw_documents=all_data.data)

    """
    export the feature names
    """
    feature_definition_file = preprocessing + '_feature_definition.data'

    feature_definition_file_handle = open(feature_definition_file, 'w', encoding='utf-8') 
    id = 0
    for name in count_vectorizer.get_feature_names():
        id = id + 1
        feature_definition_file_handle.write("%d, %s\n"%(id, name))
    feature_definition_file_handle.close()

    # """
    # export the TF-IDF vectors table into csv file
    # """
    # results = pd.DataFrame(X.toarray(), columns=count_vectorizer.get_feature_names())

    # results.to_csv(preprocessing+"1.csv", encoding='utf-8')
    # #--------------------------------------------------------------


    print("n_samples: %d, n_features: %d" % X.shape)

    print("Performing dimensionality reduction using LSA")

    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.


    svd = TruncatedSVD(n_components=200)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)


    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

    # print(X.shape)

    # """
    # export the TF-IDF vectors table into csv file
    # """
    # results = pd.DataFrame(X)

    # results.to_csv(preprocessing + "2.csv", encoding='utf-8')
    # #--------------------------------------------------------------






    """
    k-means clustering 
    """

    km = KMeans(n_clusters=k_clusters, random_state=42)
    km.fit(X)
    # clusters = km.predict(X)


    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    centers_cluster_file = preprocessing + "_centers.data"
    centers_cluster_handle = open(centers_cluster_file, 'w', encoding='utf-8') 
    id = 0
    for center in order_centroids:
        id += 1
        centers_cluster_handle.write("%d, %s\n"%(id, center))
    centers_cluster_handle.close()


    terms = count_vectorizer.get_feature_names()
    #--------------------------------------------------------------

    labels = km.predict(X);
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(X.T[0], X.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.show()


    """
    output the filenames of each clusters, order by distance from center to each sample vector
    """
    clustering_result_file = preprocessing + '_clustering_result.data'
    clustering_file_handle = open(clustering_result_file, 'w', encoding='utf-8') 

    """
    show the labels of cluster data
    """

    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]
    cluster_labels_list = []

    for i in range(0, k_clusters):
        d = km.transform(X)[:,i]
        cluster_label = []
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            cluster_label.append(terms[ind])
        print(cluster_label)
        clustering_file_handle.write("cluster %d, label %s\n"%(i, cluster_label))
        distances = []                      # distance list for each cluster
        fileindexs = []                     # filename index for each cluster
        counts_of_cluster = 0               # sample(file) count of each cluster

        for ind in range(len(labels)):
            if i == km.labels_[ind]:
                distances.append(d[ind])
                fileindexs.append(ind)
                counts_of_cluster += 1

        print('Cluster %d:' % i)
        print('counts_of_cluster: %d' % counts_of_cluster)
        
        # sort by distance and get sorted index of distance list
        indexs = np.argsort(distances)[::-1][:counts_of_cluster]

        # export filenames for each cluster
        for index in indexs:
            clustering_file_handle.write("\t %s\n"%(all_data.filenames[fileindexs[index]]))

    clustering_file_handle.close()

# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, km.labels_, sample_size=1000))



# """
# get the query article
# and predict the cluster of query article
# and display the distance array
# """
# testfile = sys.argv[1]
# testfile_handle = open(testfile, 'r') 
# lines_for_predicting = testfile_handle.read()

# lines_for_predicting = [lines_for_predicting]

# sample = count_vectorizer.transform(lines_for_predicting)
# print(sample.shape)
# sample = lsa.transform(sample)
# print(sample.shape)
# prelabel = km.predict(sample)
# distance = km.transform(sample)

# print(prelabel)
# print(distance)
dataSetDir2 = os.path.join(os.getcwd(), "dataset")
kmeans_clustering(dataSetDir2, "lemmatizing and removing 50 tmq words")

# dataSetDir2 = os.path.join(os.getcwd(), "dataset_stemming")
# kmeans_clustering(dataSetDir2, "stemming")

# dataSetDir2 = os.path.join(os.getcwd(), "dataset_lemmatizing")
# kmeans_clustering(dataSetDir2, "lemmatizing")
