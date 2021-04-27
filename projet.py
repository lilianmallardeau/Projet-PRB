#!/usr/bin/env python3

# Pattern recognition and Biometrics project
# Lilian MALLARDEAU and Vivien GAGLIANO
# 4th semester - 2021 - ENSIIE


# %% Importing libraries
import io, re
import math
from contextlib import redirect_stdout
import random
import numpy
import pandas
from matplotlib import pyplot
# import PIL.Image
from sklearn.cluster import KMeans


# %% Importing data
train_data = pandas.read_csv("data/optdigits.tra", header=None)
test_data = pandas.read_csv("data/optdigits.tes", header=None)


# %% Formatting data
train_data_x = train_data.iloc[:, :-1] # DataFrame
train_data_y = train_data.iloc[:, -1] # Series
test_data_x = test_data.iloc[:, :-1] # DataFrame
test_data_y = test_data.iloc[:, -1] # Series


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
for i in [random.randint(0, len(train_data_x)) for i in range(4)]:
    show_digit(train_data_x, i)


# %% K-means with sklearn
K = 10
kmeans = KMeans(n_clusters=K, verbose=1) #, n_init=1)
inertia_list = extract_inertia_values(kmeans.fit, train_data_x)

# %% Plotting inertia
argmin = numpy.argmin([i[-1][1] for i in inertia_list])
pyplot.plot([i[1] for i in inertia_list[argmin]])
pyplot.title("Inertia for each iteration")
pyplot.xlabel("Iteration")
pyplot.ylabel("Inertia")
pyplot.show()


# %% Histograms
for cluster_num in numpy.unique(kmeans.labels_):
    pyplot.hist(train_data_y[kmeans.labels_ == cluster_num], bins=range(10), align='left')
    pyplot.title(f"Cluster {cluster_num}")
    pyplot.xlabel("Digits")
    pyplot.ylabel("Count")
    pyplot.xlim(xmin=-0.5, xmax=9.5)
    pyplot.xticks(range(10))
    pyplot.show()

# %% Silhouette index : Mesurer la qualité du Clustering avec l’indice de la Silhouette

# Compute matrix of distances between points
point_distances = numpy.zeros(train_data_x.shape)
for i in range(train_data_x.shape[0]):
    for j in range(i):
        dist = 0
        for l in range(64):
            dist += (train_data_x[i, l] - train_data_x[j, l])**2
        point_distances[i, j] = point_distances[j, i] = numpy.sqrt(dist)

# Compute array of distances between point at given index and points of other clusters
def compute_cluster_distances(index):
    distances = []
    for i in range(kmeans.labels.len):
        distances[kmeans.labels[i]].append(point_distances[index, i])
    mean_distances = []
    for arr in distances:
        mean_distances.append(sum(arr) / arr.len)
    return mean_distances

# Compute silhouette for point at given index
def point_silhouette(index):
    cluster_distances = compute_cluster_distances(index)
    a_i = cluster_distances(kmeans.labels[index]) # mean distance between index point and other points of its cluster
    b_i = math.inf # minimum of mean distances between index point and other clusters' points
    for dist in [cluster_distances[j] for j in range(cluster_distances.len) if j != kmeans.labels[index]]: # we read through the cluster mean distance array, and skip the point's own cluster
        b_i = math.min(b_i, dist)
    return (b_i - a_i) / math.max(a_i, b_i)

# Compute the clustering's overall silhouette
silhouette = sum([point_silhouette(i) for i in range(kmeans.labels.len)]) / kmeans.labels.len

print(silhouette)
# %%