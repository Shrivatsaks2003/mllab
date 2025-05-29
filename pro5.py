import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Generate 100 random values
X = np.random.rand(100, 1)

# Label first 50 values
y = np.array([1 if i <= 0.5 else 2 for i in X[:50]])

# Plotting setup
plt.figure(figsize=(12, 8))

# Train KNN and classify remaining 50 for k = 1,2,3,4,5,20,30
for idx, k in enumerate([1, 2, 3, 4, 5, 20, 30]):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X[:50], y)
    y_pred = knn.predict(X[50:])
    
    plt.subplot(3, 3, idx + 1)
    plt.title(f"k = {k}")
    plt.scatter(X[:50], y, c=y, cmap='coolwarm', label="Train")
    plt.scatter(X[50:], y_pred, c=y_pred, marker='x', cmap='coolwarm', label="Predicted")
    plt.xlabel("X")
    plt.ylabel("Class")
    plt.legend()

# Save the plot as an image
plt.tight_layout()
plt.savefig("knn_results.png")  # Save the plot as an image
print("Plot saved as knn_results.png")
