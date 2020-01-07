from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer


from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import os
def elbow_kmeans(datasetDir, flag):
    all_data = datasets.load_files(datasetDir, 
        description=None, load_content=True, encoding='utf-8', shuffle=False)
    #--------------------------------------------------------------

    count_vectorizer = TfidfVectorizer(stop_words='english')

    X = count_vectorizer.fit_transform(raw_documents=all_data.data)

    svd = TruncatedSVD(n_components=200)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)


    # Instantiate the clustering model and visualizer



    title = "Elbow calinski_harabasz Score of Kmeans clustering"
    kwargs = {'title': title}

    plot_Dir = os.path.join(os.getcwd(), "elbow for kmeans")
    if not (os.path.exists(plot_Dir)):
            os.makedirs(plot_Dir)          
   
    filename = plot_Dir + "\\"

    model = KMeans(random_state=42)
    if flag == 0:
        suffix = "_full.png"
    if flag == 1:
        suffix = "_stemming.png"
    if flag == 2:
        suffix = "_lemmatizing.png"

    visualizer = KElbowVisualizer(
        model, k=(4,30), metric='calinski_harabasz',  **kwargs
    )

    visualizer.fit(X)        # Fit the data to the visualizer
    visualizer.set_title(title=title)
    plt.savefig(filename + title + suffix)
    plt.close()

    # visualizer.show()        # Finalize and render the figure
    # plt.savefig(filename + title + suffix)

    title = "Elbow silhouette Score of Kmeans clustering"
    kwargs = {'title': title}

    visualizer = KElbowVisualizer(
        model, k=(4,30), metric='silhouette',  **kwargs
    )
    visualizer.fit(X)        # Fit the data to the visualizer
    visualizer.set_title(title=title)
    plt.savefig(filename + title + suffix)
    plt.close()
    # visualizer.show()        # Finalize and render the figure
    # plt.savefig(filename + title + suffix)

    title = "Elbow distortion of Kmeans clustering"
    kwargs = {'title': title}

    visualizer = KElbowVisualizer(
        model, k=(4,20), metric='distortion',  **kwargs
    )
    visualizer.fit(X)        # Fit the data to the visualizer
    visualizer.set_title(title=title)
    plt.savefig(filename + title + suffix)
    plt.close()
    # visualizer.show()        # Finalize and render the figure
    # plt.savefig(filename + title + suffix)

    sil = []
    for n_cluster in range(4, 30):
        model = KMeans(random_state=42, n_clusters=n_cluster).fit(X)
        labels = model.labels_
        sil.append(silhouette_score(X, labels, metric = 'euclidean'))
        # model = KMeans(random_state=42, n_clusters=n_cluster)
        # Svisualizer = SilhouetteVisualizer(model)
        # Svisualizer.fit(X)    # Fit the data to the visualizer
        # Svisualizer.poof()    # Draw/show/poof the data
        # plt.
    plt.plot(list(range(4, 30)), sil)
    plt.grid(True)
    plt.savefig(filename + "sihouette" + suffix)
    plt.close()
    # plt.show()

# full_dataSetDir = os.path.join(os.getcwd(), "dataset_full")
# stemming_datasetDir = os.path.join(os.getcwd(), "dataset_stemming")
lemmatizing_datasetDir = os.path.join(os.getcwd(), "dataset_lemmatizing")

# elbow_kmeans(full_dataSetDir, 0)
# elbow_kmeans(stemming_datasetDir, 1)
elbow_kmeans(lemmatizing_datasetDir, 2)


