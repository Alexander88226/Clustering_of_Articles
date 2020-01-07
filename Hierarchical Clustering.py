import os
import sys
import collections
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import dump_svmlight_file


import time
import string


from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from scipy.cluster.hierarchy import dendrogram
import scipy.cluster.hierarchy as shc
from sklearn.metrics import silhouette_score
import nltk
from nltk.corpus import words
nltk.download('words')

remaining_word_list = ['blockchain', 'crowdfunding']

"""
set parameters for plot
"""
plot_kwds = {'alpha' : 0.8, 's' : 80, 'linewidths':0}
n_clusters = 10
top_word_count = 50

ClusteringDir2 = os.path.join(os.getcwd(), "Hierarchical Clustering")
if not (os.path.exists(ClusteringDir2)):
    os.makedirs(ClusteringDir2)          

"""
define function plot_cluster
description: display feature distribution for each cluster
"""
def plot_clusters(data, algorithm, preprocessing, distance, args, kwds):

    prefix = ClusteringDir2 + "\\" + preprocessing + "_" + distance
    start_time = time.time()
    algorithm(*args, **kwds)
    labels = algorithm(*args, **kwds).fit_predict(data)

    training_data_file = prefix + '_hierarchical_training_data_file.data'
    dump_svmlight_file(data, labels, training_data_file)

    return labels

    # end_time = time.time()
    # palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    # colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    # plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    # frame = plt.gca()
    # frame.axes.get_xaxis().set_visible(False)
    # frame.axes.get_yaxis().set_visible(False)

    # plt.title(preprocessing + ' Clusters found by {}'.format(str(algorithm.__name__)), fontsize=14)
    # plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
    # # plt.show()
    # plt.savefig(prefix + "_hierarchical clustering.png")
    # plt.close()

"""
define fucntion get_top_n_words
description: get n top frequent words
"""


def get_top_n_words_n_que(corpus, n=None):

    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    word_count = 0
    words_freq_que = []
    for word, idx in vec.vocabulary_.items():
        if word in words.words() or word in remaining_word_list:
            word_count += 1
            words_freq_que.append((word, sum_words[0, idx]))
        if word_count >= n:
            break
    # words_freq_que = [(word, sum_words[0, idx]) if word in words.words() for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq_que, key = lambda x: x[1], reverse=True)
    return words_freq[:n], words_freq_que[:n]


def hierarchical_clustering(datasetDir, preprocessing, distance):


    all_data = datasets.load_files(datasetDir, description=None, load_content=True, encoding='utf-8', shuffle=False)

    prefix = ClusteringDir2 + "\\" + preprocessing + "_" + distance
    """
    Apply Tf-idf vectorizer with stop words
    """
    count_vectorizer = TfidfVectorizer(stop_words='english')

    """
    Learn vocabulary and tf-idf, return term-document matrix.
    """
    X = count_vectorizer.fit_transform(raw_documents=all_data.data).toarray()

    """
    Apply Dimensionality reduction using truncated SVD (aka LSA).
    """
    svd = TruncatedSVD(n_components=200)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    data = lsa.fit_transform(X)

    labels = plot_clusters(data, cluster.AgglomerativeClustering, preprocessing, distance, (), {'n_clusters':n_clusters, 'linkage':'complete', 'affinity':distance})
  
    clustering_file = prefix + '_clustering_result.data'
    clustering_file_handle = open(clustering_file, 'w', encoding='utf-8') 

    
    for index in range(0, 10):
        sub_corpus = []
        corpus_files = []
        # cluster_label = []
        for key, label in enumerate(labels):
            if index == label:
                sub_corpus.append(all_data.data[key])
                corpus_files.append(all_data.filenames[key])
        
        top_n_words,_ = get_top_n_words_n_que(sub_corpus, 10)
        top_n_words = np.array(top_n_words)

        clustering_file_handle.write("cluster %d, label %s\n"%(index, list(top_n_words[:,0])))
        for file in corpus_files:
            clustering_file_handle.write("\t %s\n" % file)
# print(all_data.filenames[key])

    # Plot silhouette score
    sil = []
    for n_cluster in range(4, 30):
        model = cluster.AgglomerativeClustering(n_clusters=n_cluster, affinity=distance, linkage='complete').fit(X)
        labels = model.labels_
        model.__dict__
        sil.append(silhouette_score(X, labels, metric = 'euclidean'))
        # model = KMeans(random_state=42, n_clusters=n_cluster)
        # Svisualizer = SilhouetteVisualizer(model)
        # Svisualizer.fit(X)    # Fit the data to the visualizer
        # Svisualizer.poof()    # Draw/show/poof the data
        # plt.
    plt.plot(list(range(4, 30)), sil)
    plt.grid(True)
    plt.savefig(prefix + "_sihouette_score.png")
    plt.close()

    """
    Plot the hierarchical clustering as a dendrogram
    """

    plt.figure(figsize=(10, 7))
    plt.title("Hierarchical Clustering Dendograms")
    dend = shc.dendrogram(shc.linkage(data, method='complete'))
    plt.savefig(prefix + "_hierarchical clustering_Dendograms.png")
    plt.close()


    top_n_words, que_n_words = get_top_n_words_n_que(all_data.data, top_word_count);

    """
    Output n top frequent words into file
    """

    top_n_words_file = prefix + "top_n_words.data"
    out_filepath_handle = open(top_n_words_file, "w")
    word_names = []
    word_freqs = []
    reverse_freqs = []
    for word in top_n_words:
        word_names.append(word[0])
        word_freqs.append(word[1])
        reverse_freqs.append(word[1])
        out_filepath_handle.write(str(word)+'\n')
    out_filepath_handle.close()

    """
    visualize the n top frequent words 
    """
    index = np.arange(0, top_word_count)
    reverse_freqs.reverse()
    word_names.reverse()
    plt.barh(index, reverse_freqs)
    plt.yticks(index, word_names)
    plt.title(str(top_word_count) + " top frequent words")
    plt.ylabel("words")
    plt.xlabel("frequency")
    plt.savefig(prefix + "_" + str(top_word_count) + " top frequent words.png")
    plt.close()


    word_freqs = np.array(reverse_freqs)
    cooccurrence_matrix = np.outer(word_freqs, word_freqs)

    ax = sns.heatmap(cooccurrence_matrix, linewidth=0.1)
    plt.yticks(index, word_names, rotation='horizontal')
    plt.xticks(index, word_names, rotation = 'vertical')
    plt.title("Words Co-occurrence")
    plt.savefig(prefix + "_Co-occurrence.png")
    plt.close()

# dataSetDir2 = os.path.join(os.getcwd(), "dataset_full")
# hierarchical_clustering(dataSetDir2, "full preprocessing")

# dataSetDir2 = os.path.join(os.getcwd(), "dataset_stemming")
# hierarchical_clustering(dataSetDir2, "stemming")

dataSetDir2 = os.path.join(os.getcwd(), "dataset_lemmatizing")
hierarchical_clustering(dataSetDir2, "lemmatizing", "euclidean")
hierarchical_clustering(dataSetDir2, "lemmatizing", "cosine")
hierarchical_clustering(dataSetDir2, "lemmatizing", "manhattan")
hierarchical_clustering(dataSetDir2, "lemmatizing", "l2")
hierarchical_clustering(dataSetDir2, "lemmatizing", "l1")
hierarchical_clustering(dataSetDir2, "lemmatizing", "precomputed")
hierarchical_clustering(dataSetDir2, "lemmatizing", "cityblock")