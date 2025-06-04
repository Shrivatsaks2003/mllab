import pyforest

X = np.random.rand(100, 1)
y = np.where(X[:50] <= 0.5, 1, 2)

plt.figure(figsize=(10, 6))
for i, k in enumerate([1, 2, 3, 4, 5, 20, 30]):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X[:50], y.ravel())
    pred = model.predict(X[50:])
    
    plt.subplot(3, 3, i + 1)
    plt.title(f"k = {k}")
    plt.scatter(X[:50], y, c=y, cmap='coolwarm', label="Train")
    plt.scatter(X[50:], pred, c=pred, marker='x', cmap='coolwarm', label="Predicted")
    plt.legend()

plt.tight_layout()
plt.show()
