import pyforest

# Load and split data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)

# Evaluate
print(f"Model Accuracy: {accuracy_score(y_test, clf.predict(X_test)) * 100:.2f}%")

# Predict new sample
prediction = clf.predict([X_test[0]])
print(f"Predicted Class: {'Benign' if prediction == 1 else 'Malignant'}")

# Plot tree
plt.figure(figsize=(12,8))
tree.plot_tree(clf, filled=True, feature_names=load_breast_cancer().feature_names, class_names=load_breast_cancer().target_names)
plt.title("Decision Tree - Breast Cancer Dataset")
plt.show()
