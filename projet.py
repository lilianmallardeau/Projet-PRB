#!/usr/bin/env python3

# Pattern recognition and Biometrics project
# Lilian MALLARDEAU and Vivien GAGLIANO
# 4th semester - 2021 - ENSIIE


# %% Importing libraries
import io, re
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
def compute_point_distances():
    distances = numpy.zeros(train_data_x.shape)
    for i in range(train_data_x.shape[0]):
        for j in range(i):
            dist = 0
            for l in range(64):
                dist += (train_data_x[i, l] - train_data_x[j, l])**2
            distances[i, j] = distances[j, i] = numpy.sqrt(dist)
    return distances

# Compute array of distances between point at given index and points of other clusters
def compute_cluster_distances(index):
    point_distances = compute_point_distances(train_data_x)
    distances = []
    for i in range(K):

# J'EN ETAIS LA




def silhouette():
    distances = compute_point_distances(train_data_x)
    sil = []
    for i in range(kmeans.labels.len):
        same_cluster = []
        diff_cluster = []
        for j in range(kmeans.labels.len):
            if i == j:
                break
            if kmeans.labels[i] == kmeans.labels[j]:
                same_cluster.append(distances[j])
            else:
                diff_cluster.append(distances[j])
        # distance moyenne entre i et les points de son cluster
        a_i = sum(same_cluster) / same_cluster.len
        b_i = # minimum des distances moyennes entre i et les points de chaque autre cluster
        sil[i] = (b_i - a_i) / max(a_i, b_i)

    return


# %%
