from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot import pyforest

X, y = load_breast_cancer(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)
y_kmeans = KMeans(n_clusters=2, random_state=42).fit_predict(X_scaled)

print(confusion_matrix(y, y_kmeans), '\n', classification_report(y, y_kmeans))

pca = PCA(n_components=2)
df = pd.DataFrame(pca.fit_transform(X_scaled), columns=['PC1', 'PC2'])
df['Cluster'], df['True'] = y_kmeans, y

for col, title, palette in [('Cluster', 'K-Means Clustering', 'Set1'), ('True', 'True Labels', 'coolwarm')]:
    sns.scatterplot(data=df, x='PC1', y='PC2', hue=col, palette=palette, s=100, edgecolor='black')
    plt.title(title); plt.show()

sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set1', s=100, edgecolor='black')
centers = pca.transform(KMeans(n_clusters=2, random_state=42).fit(X_scaled).cluster_centers_)
plt.scatter(*centers.T, c='red', marker='X', s=200, label='Centroids')
plt.title('Clusters with Centroids'); plt.legend(); plt.show()
as plt
import seaborn as sns

# Load and scale data
X, y = load_breast_cancer(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Evaluate clustering
print(confusion_matrix(y, y_kmeans), '\n')
print(classification_report(y, y_kmeans))

# PCA for visualization
pca = PCA(n_components=2)
df = pd.DataFrame(pca.fit_transform(X_scaled), columns=['PC1', 'PC2'])
df['Cluster'] = y_kmeans
df['True'] = y

# Plot clustering and true labels
for col, title, palette in [('Cluster', 'K-Means Clustering', 'Set1'), ('True', 'True Labels', 'coolwarm')]:
    sns.scatterplot(data=df, x='PC1', y='PC2', hue=col, palette=palette, s=100, edgecolor='black')
    plt.title(title)
    plt.show()

# Plot clusters with centroids
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set1', s=100, edgecolor='black')
centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(*centers.T, c='red', marker='X', s=200, label='Centroids')
plt.title('Clusters with Centroids')
plt.legend()
plt.show()
