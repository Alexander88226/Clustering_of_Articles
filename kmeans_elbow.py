from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer


from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

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
    model = KMeans(random_state=42)
    if flag == 0:
        title = "Elbow Score of Kmeans clustering for full preprocessing(stemming and lemmatizing)"
    if flag == 1:
        title = "Elbow Score of Kmeans clustering for stemming"
    if flag == 2:
        title = "Elbow Score of Kmeans clustering for lemmatizing"

    kwargs = {'title': title}

    visualizer = KElbowVisualizer(
        model, k=(4,20), metric='calinski_harabasz', timings=False, locate_elbow=False, **kwargs
    )

    visualizer.fit(X)        # Fit the data to the visualizer

    visualizer.set_title(title=title)
    visualizer.show()        # Finalize and render the figure

full_dataSetDir = os.path.join(os.getcwd(), "dataset")
stemming_datasetDir = os.path.join(os.getcwd(), "dataset_stemming")
lemmatizing_datasetDir = os.path.join(os.getcwd(), "dataset_lemmatizing")

elbow_kmeans(full_dataSetDir, 0)
elbow_kmeans(stemming_datasetDir, 1)
elbow_kmeans(lemmatizing_datasetDir, 2)


