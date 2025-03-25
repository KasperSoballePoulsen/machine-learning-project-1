# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 11:23:02 2018

@author: sila
"""


from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from matplotlib import pyplot
from pandas import DataFrame
# generate 2d classification dataset
# generate 2d classification dataset
X, y = make_circles(n_samples=100, noise=0.05)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)

train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")


xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.1, X[:,0].max()+0.1, 200),
                     np.linspace(X[:,1].min()-0.1, X[:,1].max()+0.1, 200))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  # Predict labels for meshgrid points
Z = Z.reshape(xx.shape)  # Reshape for contour plotting

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')

# Plot training data
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', edgecolors='k', marker='o', label="Train Data")

# Plot test data
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', edgecolors='k', marker='s', label="Test Data")

# Labels and title
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("sigmoid")
plt.show()