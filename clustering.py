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
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 
dataSetDir2 = os.path.join(os.getcwd(), "dataset")

# """
# filter non-ascii string, for instance //x08
# """
# for subdir, dirs, files in os.walk(dataSetDir2):
#     for file in files:
#         subdir_replaced = subdir.replace("\\", "/")     # Set all seperators to "/"
#         filepath = subdir_replaced + "/" + file         # full org dataset file path
#         subdirOnly = subdir_replaced[len(dataSetDir2):]  # Get only sub directory names
#         extractedFilePath = dataSetDir2 + subdirOnly
#         extractedFileName = extractedFilePath + "/" + file

#         if not (os.path.exists(extractedFilePath)):
#                 os.makedirs(extractedFilePath)          # Create sub directory in dataSetDir2
#         orgDatasetFile = open(filepath, 'r')            # Open org dataset file
#         fileLines = orgDatasetFile.readlines()          # read text file as line array
#         extractedFile = open(extractedFileName, 'w')    # Create new file for writing subject and body

        
#  load our data-----------------------------------------
all_data = datasets.load_files(dataSetDir2, 
    description=None, load_content=True, encoding='utf-8', shuffle=False)
#--------------------------------------------------------------
labels = all_data.target


"""
This is a function that will count the number of times each word in the
dataset occurs and project that count into a vector(TF-IDF vector). 
"""
count_vectorizer = TfidfVectorizer(stop_words='english')

X = count_vectorizer.fit_transform(raw_documents=all_data.data)

"""
export the feature names
"""
feature_definition_file = 'feature_definition.data'

feature_definition_file_handle = open(feature_definition_file, 'w') 
id = 0
for name in count_vectorizer.get_feature_names():
    id = id + 1
    feature_definition_file_handle.write("%d, %s\n"%(id, name))
feature_definition_file_handle.close()

"""
export the TF-IDF vectors table into csv file
"""
results = pd.DataFrame(X.toarray(), columns=count_vectorizer.get_feature_names())

results.to_csv("1.csv", encoding='utf-8')
#--------------------------------------------------------------


print("n_samples: %d, n_features: %d" % X.shape)

print("Performing dimensionality reduction using LSA")

# Vectorizer results are normalized, which makes KMeans behave as
# spherical k-means for better results. Since LSA/SVD results are
# not normalized, we have to redo the normalization.

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

svd = TruncatedSVD(n_components=200)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)


explained_variance = svd.explained_variance_ratio_.sum()
print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

print(X.shape)

"""
export the TF-IDF vectors table into csv file
"""
results = pd.DataFrame(X)

results.to_csv("2.csv", encoding='utf-8')
#--------------------------------------------------------------


"""
Finding Optimal Clusters
"""

# def find_optimal_clusters(data, max_k):
#     iters = range(2, max_k+1, 2)
    
#     sse = []
#     for k in iters:
#         sse.append(KMeans(n_clusters=k).fit(data).inertia_)#, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_
#         print('Fit {} clusters'.format(k))
#     f, ax = plt.subplots(1, 1)
#     ax.plot(iters, sse, marker='o')
#     ax.set_xlabel('Cluster Centers')
#     ax.set_xticks(iters)
#     ax.set_xticklabels(iters)
#     ax.set_ylabel('SSE')
#     ax.set_title('SSE by Cluster Center Plot')
    
# find_optimal_clusters(X, 20)
# plt.show()
#--------------------------------------------------------------



"""
k-means clustering 
"""

k_clusters = 10
km = KMeans(n_clusters=k_clusters, random_state=3425)
km.fit(X)
# clusters = km.predict(X)


order_centroids = km.cluster_centers_.argsort()[:, ::-1]
centers_cluster_file = "centers.data"
centers_cluster_handle = open(centers_cluster_file, 'w') 
id = 0
for center in order_centroids:
    id += 1
    centers_cluster_handle.write("%d, %s\n"%(id, center))
centers_cluster_handle.close()


terms = count_vectorizer.get_feature_names()
#--------------------------------------------------------------

"""
Plotting Clusters

"""
# def plot_tsne_pca(data, labels):
#     max_label = max(labels)
#     print(max_label)
#     max_items = np.random.choice(range(data.shape[0]), size=100, replace=False)
    
#     pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())
#     tsne = TSNE().fit_transform(PCA(n_components=100).fit_transform(data[max_items,:].todense()))
    
    
#     idx = np.random.choice(range(pca.shape[0]), size=100, replace=False)
#     label_subset = labels[max_items]
#     label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
    
#     f, ax = plt.subplots(1, 2, figsize=(14, 6))
    
#     ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
#     ax[0].set_title('PCA Cluster Plot')
    
#     ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
#     ax[1].set_title('TSNE Cluster Plot')
    
# plot_tsne_pca(X, clusters)
# plt.show()

"""
output the filenames of each clusters, order by distance from center to each sample vector
"""
clustering_result_file = 'clustering_result.data'
clustering_file_handle = open(clustering_result_file, 'w') 

"""
show the labels of cluster data
"""
print(km.labels_)

for i in range(0, k_clusters):
    d = km.transform(X)[:,i]
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
        clustering_file_handle.write("%d, %s\n"%(i, all_data.filenames[fileindexs[index]]))

clustering_file_handle.close()



print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))
