from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
X, y, names = iris.data, iris.target, iris.target_names
X_pca = PCA(2).fit_transform(X)
colors = ['r', 'g', 'b']

for i in range(3):
    plt.scatter(X_pca[y==i, 0], X_pca[y==i, 1], c=colors[i], label=names[i])

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Iris Dataset')
plt.legend()
plt.grid()
plt.show()
