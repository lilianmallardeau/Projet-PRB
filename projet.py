#!/usr/bin/env python3

# Pattern recognition and Biometrics project
# Lilian MALLARDEAU and Vivien GAGLIANO
# 4th semester - 2021 - ENSIIE


# %% Importing libraries
import io
import random
import re
from collections import Counter
from contextlib import redirect_stdout

import numpy
import pandas
import seaborn
# import PIL.Image
from matplotlib import pyplot
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score

# %% Parameters
PLOT_RANDOM_DIGITS = False
PLOT_INERTIA = True
PLOT_HISTOGRAMS = True

# %% Importing data
train_data = pandas.read_csv("data/optdigits.tra", header=None)
test_data = pandas.read_csv("data/optdigits.tes", header=None)

# %% Formatting data
train_data_x = train_data.iloc[:, :-1]  # DataFrame
train_data_y = train_data.iloc[:, -1]  # Series
test_data_x = test_data.iloc[:, :-1]  # DataFrame
test_data_y = test_data.iloc[:, -1]  # Series


# %% Defining some useful functions
def show_digit(dataset, index):
    digit = dataset.iloc[index, :].to_numpy().reshape(8, 8)
    pyplot.imshow(digit, 'Greys')
    pyplot.show()
    # PIL.Image.fromarray(digit).show()


def intercept_output(function, *args, **kwargs):
    f = io.StringIO()
    with redirect_stdout(f):
        result = function(*args, **kwargs)
    return result, f.getvalue()


def extract_inertia_values(function, *args, **kwargs):
    result, output = intercept_output(function, *args, **kwargs)
    inertia_list = list()
    pattern = re.compile(r"^Iteration (\d+), inertia (\d+.\d+)$")
    for line in output.splitlines():
        if line.startswith('Iteration'):
            iteration = pattern.match(line)
            inertia_list[-1].append((int(iteration.group(1)), float(iteration.group(2))))
        elif line.startswith('Initialization complete'):
            inertia_list.append(list())
    return inertia_list


# %% Showing some random digits from train dataset
if PLOT_RANDOM_DIGITS:
    for i in [random.randint(0, len(train_data_x)) for i in range(4)]:
        show_digit(train_data_x, i)

########################################################################################################################
# %% K-means with sklearn
K = 10
kmeans = KMeans(n_clusters=K, verbose=1)
inertia_list = extract_inertia_values(kmeans.fit, train_data_x)

# %% Plotting inertia
if PLOT_INERTIA:
    argmin = numpy.argmin([i[-1][1] for i in inertia_list])
    pyplot.figure(figsize=(10, 5))
    pyplot.plot([i[1] for i in inertia_list[argmin]])
    pyplot.title("Inertia at each iteration")
    pyplot.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    pyplot.xlabel("Iteration")
    pyplot.ylabel("Inertia")
    pyplot.show()


# %% Histograms
def plot_histograms(clustering, title=None):
    fig = pyplot.figure(figsize=(10, 7.5))
    fig.suptitle(title)
    gs = GridSpec(4, 3)
    for cluster_num in numpy.unique(clustering.labels_):
        ax = fig.add_subplot(gs[cluster_num])
        ax.hist(train_data_y[clustering.labels_ == cluster_num], bins=range(11), align='left')
        ax.set_title(f"Cluster {cluster_num}", fontsize=10)
        # ax.set_xlabel("Digits")
        # ax.set_ylabel("Count")
        ax.set_xlim(xmin=-0.5, xmax=9.5)
        ax.set_xticks(range(10))
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.show()


if PLOT_HISTOGRAMS:
    plot_histograms(kmeans, "KMeans clustering")


# %% Silhouette index : Mesurer la qualit?? du Clustering avec l???indice de la Silhouette

# # Compute matrix of distances between points
# point_distances = numpy.zeros((train_data_x.shape[0], train_data_x.shape[0]))
# for i in range(train_data_x.shape[0]):
#     for j in range(i):
#         dist = 0
#         for l in range(64):
#             dist += (train_data_x.iloc[i, l] - train_data_x.iloc[j, l])**2
#         point_distances[i, j] = point_distances[j, i] = numpy.sqrt(dist)
#
# # Compute array of distances between point at given index and points of other clusters
# def compute_cluster_distances(index):
#     distances = []
#     for i in range(kmeans.labels.len):
#         distances[kmeans.labels[i]].append(point_distances[index, i])
#     mean_distances = []
#     for arr in distances:
#         mean_distances.append(sum(arr) / arr.len)
#     return mean_distances
#
# # Compute silhouette for point at given index
# def point_silhouette(index):
#     cluster_distances = compute_cluster_distances(index)
#     a_i = cluster_distances(kmeans.labels[index]) # mean distance between index point and other points of its cluster
#     b_i = math.inf # minimum of mean distances between index point and other clusters' points
#     for dist in [cluster_distances[j] for j in range(cluster_distances.len) if j != kmeans.labels[index]]: # we read through the cluster mean distance array, and skip the point's own cluster
#         b_i = math.min(b_i, dist)
#     return (b_i - a_i) / math.max(a_i, b_i)
#
# # Compute the clustering's overall silhouette
# silhouette = sum([point_silhouette(i) for i in range(kmeans.labels.len)]) / kmeans.labels.len
#
# print(silhouette)


def silhouette(clustering):
    return silhouette_score(train_data_x, clustering.labels_)


print(f"KMeans silhouette for K={K}: {silhouette(kmeans)}")

# %% Selecting best clustering from K=10 to 15
K_range = range(10, 16)
print(f"Computing KMeans for K={K_range.start} to {K_range.stop - 1}...")
clusterings = {K: KMeans(n_clusters=K).fit(train_data_x) for K in K_range}
# [plot_histograms(c, f"KMeans clustering for {K} clusters") for K, c in clusterings.items()]
silhouette_values = {K: silhouette(clusterings[K]) for K in K_range}

print("Computing silhouette index for each KMeans...")
best_K = max(silhouette_values, key=silhouette_values.get)
best_clustering = clusterings[best_K]
print(f"Best clustering for K={best_K}, silhouette={silhouette_values[best_K]}")


# %% Testing
def perform_testing(clustering, print_conf_mat=False):
    labels = {cluster_num: Counter(train_data_y[clustering.labels_ == cluster_num]).most_common(1)[0][0] for
              cluster_num in numpy.unique(clustering.labels_)}

    # Check that all the digits are represented
    print(f"K={clustering.n_clusters}, Number of different labels represented:",
          len(numpy.unique(list(labels.values()))))

    # Computing predicted labels for test set
    predicted_labels = [labels[found_cluster] for found_cluster in clustering.predict(test_data_x)]
    # print(predicted_labels == test_data_y)

    # Plot confusion matrix
    if print_conf_mat:
        plot_conf_mat(predicted_labels, f"KMeans confusion matrix for K={clustering.n_clusters}")

    # Return accuracy
    return accuracy_score(test_data_y, predicted_labels, normalize=True)


# Assigning label to each cluster, by majority vote, and computing accuracy
accuracy = {K: 100 * perform_testing(clustering) for K, clustering in clusterings.items()}


# %% Confusion matrix
def plot_conf_mat(pred, title):
    conf_mat_kmeans = confusion_matrix(test_data_y, pred)
    print(title)
    print(conf_mat_kmeans)
    seaborn.heatmap(conf_mat_kmeans, annot=True, fmt='d', cmap="YlGnBu").set_title("a")
    pyplot.title(title)
    pyplot.show()


perform_testing(best_clustering, True)

# %% Let's plot silhouette index and accuracy in the same plot
fig, ax1 = pyplot.subplots(figsize=(10, 5))
ax1.set_xlabel("Clusters number K")
ax1.set_xticks(list(clusterings.keys()))

ax1.set_ylabel("Accuracy (%)", color="tab:red")
ax1.plot(K_range, list(accuracy.values()), color="tab:red")
ax1.tick_params(axis='y', labelcolor="tab:red")

ax2 = ax1.twinx()
ax2.set_ylabel("Silhouette index", color="tab:blue")
ax2.plot(K_range, list(silhouette_values.values()), color="tab:blue")
ax2.tick_params(axis='y', labelcolor="tab:blue")

fig.tight_layout()
ax1.grid(axis="x")
fig.show()

########################################################################################################################
# %% Hierarchical clustering

print("\nComputing hierarchical clustering...")
clustering = AgglomerativeClustering(n_clusters=K, linkage='ward').fit(train_data_x)
dendrogram = sch.dendrogram(sch.linkage(train_data_x, method='ward'))
pyplot.title("Hierarchical clustering dendrogram")
pyplot.xlabel("Digits")
pyplot.ylabel("Euclidian distance")
pyplot.show()

# %% Histograms for hierarchical clustering
if PLOT_HISTOGRAMS:
    plot_histograms(clustering, "Hierarchical clustering")

# %% Comparing silhouette value from KMeans and hierarchical clusterings
print(f"Kmeans clustering silhouette for {K} clusters: {silhouette(kmeans)}")
print(f"Hierarchical clustering silhouette for {K} clusters: {silhouette(clustering)}")

# %% Finding best K
K_range = range(10, 16)
print(f"Computing Hierarchical clustering for K={K_range.start} to {K_range.stop - 1}...")
clusterings = {K: AgglomerativeClustering(n_clusters=K).fit(train_data_x) for K in K_range}
silhouette_values = {K: silhouette(clusterings[K]) for K in K_range}

print("Computing silhouette index for each clustering...")
best_K = max(silhouette_values, key=silhouette_values.get)
best_clustering = clusterings[best_K]
print(f"Best clustering for K={best_K}, silhouette={silhouette_values[best_K]}")

# %% Testing
# Assigning label to each cluster, by majority vote
labels = {cluster_num: Counter(train_data_y[best_clustering.labels_ == cluster_num]).most_common(1)[0][0] for
          cluster_num in numpy.unique(best_clustering.labels_)}

# Check that all the digits are represented
print("Number of different labels represented:", len(numpy.unique(list(labels.values()))))

# Computing centroids for each cluster (Sklean doesn't do it automatically like with KMeans)
centroids = {cluster_num: numpy.mean(train_data_x[best_clustering.labels_ == cluster_num]) for cluster_num in
             numpy.unique(best_clustering.labels_)}


# Building a classifier from hierarchical clustering
def predict(digit):
    distances = {cluster_num: numpy.sum(numpy.square(centroid - digit)) for cluster_num, centroid in centroids.items()}
    return labels[min(distances, key=distances.get)]


# Computing predicted labels for test set
predicted_labels = [predict(digit) for index, digit in test_data_x.iterrows()]
# print(predicted_labels == test_data_y)
accuracy = accuracy_score(predicted_labels, test_data_y, normalize=True)

# %% Confusion matrix
plot_conf_mat(predicted_labels, f"Hierarchical clustering confusion matrix for K={best_K}")
