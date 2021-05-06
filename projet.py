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
from matplotlib import pyplot
import seaborn
# import PIL.Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix

# %% Parameters
PLOT_RANDOM_DIGITS = True
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

# %% K-means with sklearn
K = 10
kmeans = KMeans(n_clusters=K, verbose=1)
inertia_list = extract_inertia_values(kmeans.fit, train_data_x)

# %% Plotting inertia
if PLOT_INERTIA:
    argmin = numpy.argmin([i[-1][1] for i in inertia_list])
    pyplot.plot([i[1] for i in inertia_list[argmin]])
    pyplot.title("Inertia for each iteration")
    pyplot.xlabel("Iteration")
    pyplot.ylabel("Inertia")
    pyplot.show()

# %% Histograms
if PLOT_HISTOGRAMS:
    for cluster_num in numpy.unique(kmeans.labels_):
        pyplot.hist(train_data_y[kmeans.labels_ == cluster_num], bins=range(10), align='left')
        pyplot.title(f"Cluster {cluster_num}")
        pyplot.xlabel("Digits")
        pyplot.ylabel("Count")
        pyplot.xlim(xmin=-0.5, xmax=9.5)
        pyplot.xticks(range(10))
        pyplot.show()


# %% Silhouette index : Mesurer la qualité du Clustering avec l’indice de la Silhouette

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


# print(f"Silhouette: {silhouette(kmeans)}")

# %% Selecting best clustering from K=10 to 15
K_range = range(10, 16)
print(f"Computing KMeans for K={K_range.start} to {K_range.stop - 1}...")
clusterings = {K: KMeans(n_clusters=K).fit(train_data_x) for K in K_range}
silhouette_values = {K: silhouette(clusterings[K]) for K in K_range}

print("Computing silhouette index for each KMeans...")
best_K = max(silhouette_values, key=silhouette_values.get)
best_clustering = clusterings[best_K]
print(f"Best clustering for K={best_K}, silhouette={silhouette_values[best_K]}")

# %% Testing
# Assigning label to each cluster, by majority vote
labels = {cluster_num: Counter(train_data_y[best_clustering.labels_ == cluster_num]).most_common(1)[0][0] for
          cluster_num in numpy.unique(best_clustering.labels_)}

# Check that all the digits are represented
print("Number of different labels represented:", len(numpy.unique(list(labels.values()))))

# Computing predicted labels for test set
predicted_labels = [labels[found_cluster] for found_cluster in best_clustering.predict(test_data_x)]
# print(predicted_labels == test_data_y)

# %% Confusion matrix
conf_mat = confusion_matrix(test_data_y, predicted_labels)
print(f"KMeans confusion matrix for K={best_K}")
print(conf_mat)
seaborn.heatmap(conf_mat, annot=True, fmt='d', cmap="YlGnBu").set_title("a")
pyplot.title(f"KMeans confusion matrix for K={best_K}")
pyplot.show()
