from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load data
X, y = fetch_olivetti_faces(shuffle=True, random_state=42, return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and predict
gnb = GaussianNB().fit(X_train, y_train)
y_pred = gnb.predict(X_test)

# Evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n')
print(classification_report(y_test, y_pred, zero_division=1))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"\nCross-validation Accuracy: {cross_val_score(gnb, X, y, cv=5).mean() * 100:.2f}%")

# Plot sample faces with predictions
fig, axes = plt.subplots(3, 5, figsize=(12, 8))
for ax, img, label, pred in zip(axes.ravel(), X_test[:15], y_test[:15], y_pred[:15]):
    ax.imshow(img.reshape(64, 64), cmap='gray')
    ax.set_title(f"T:{label} / P:{pred}")
    ax.axis('off')
plt.tight_layout()
plt.show()
