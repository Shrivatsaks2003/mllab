# Develop a program to implement k-means clustering using Wisconsin Breast Cancer data set and visualize
# the clustering result. 
#Alternate 10th program
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = load_breast_cancer()
x = data.data
y = data.target

kmeans = KMeans(n_clusters=2, random_state=2)
kmeans.fit(x)

cluster_labels = kmeans.labels_

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

plt.figure(figsize=(10,8))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.title('K-Means Clustering on Breast Cancer Data')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()