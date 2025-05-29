import pyforest

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
