from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target
X_pca = PCA(2).fit_transform(X)

for i in range(3):
    plt.scatter(X_pca[y==i, 0], X_pca[y==i, 1])

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Iris Dataset')

plt.grid()
plt.show()
