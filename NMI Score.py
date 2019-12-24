import os
from sklearn import datasets
from sklearn import metrics
import matplotlib.pyplot as plt

kmeans_dir =  os.path.join(os.getcwd(), "clustering_result")
hierarch_dir = os.path.join(os.getcwd(), "Hierarchical Clustering")

def NMI_Score_Estimation(dir1, dir2, preprocessing):

    kmeans_prefix = kmeans_dir + "\\" + preprocessing
    kmeans_data = kmeans_prefix + 'kmeans_training_data_file.data'
    hierarch_prefix = hierarch_dir + "\\" + preprocessing
    hierarch_data = hierarch_prefix + '_hierarchical_training_data_file.data'

    r_feature_vectors, r_targets = datasets.load_svmlight_file(kmeans_data)
    t_feature_vectors, t_targets = datasets.load_svmlight_file(hierarch_data)

    score = metrics.normalized_mutual_info_score(r_targets, t_targets)
    print(score)

NMI_Score_Estimation(kmeans_dir, hierarch_dir, "full preprocessing")
NMI_Score_Estimation(kmeans_dir, hierarch_dir, "stemming")
NMI_Score_Estimation(kmeans_dir, hierarch_dir, "lemmatizing")


