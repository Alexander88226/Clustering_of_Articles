This is readme file.

Architecture of python script files and directories, description
directories
    17 cluster
       figures for compareing clustering with 17 cluster
    corpus
       origin text corpus(no preprocessing)
    dataset
       result corpus after stemming and lemmatizing
    dataset_lemmatizing
       result corpus after lemmatizing
    dataset_stemming
       result corpus after stemming
    elbow for kmeans
       elbow curve for kmeans with 4-20 clusters
    Hierarchical Clustering
       visualiztion of top30 words frequency, hierarchical clustering, co-occurrence, dendogram for stemming, lemmatizing, and both of these two.
python script files
    clustering.py
       tf-idf vectorization, kmeans clustering, sorting the files by distance
    comparing_clustering.py
       comparing several clustering algorithms
    Hierarchical Clustering.py
       hierarchical clustering, get top n words, dendogram graph, heatmap of co-occurrence matrix
    kmeans_elbow.py
       display elbow curve for kmeans clustering
    preprocessing.py
       preprocessing of corpus including removing numbers, remove punctuation, lemmatizing, stemming