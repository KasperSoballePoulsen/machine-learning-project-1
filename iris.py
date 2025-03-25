from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

#Imports the iris dataset
iris_df = datasets.load_iris()

#Makes a PCA that reduces data into two dimensions
pca = PCA(2)

#Vertically splits the dataset into features and targets
X, y = iris_df.data, iris_df.target

#Converts the 4 dimensional X data into 2 dimensions and prints the shape
X_proj = pca.fit_transform(X)


# Set the size of the plot
plt.figure(figsize=(10, 4))

# create color map
colormap = np.array(['red', 'lime', 'black', 'blue', 'yellow', 'green', 'red'])

#Grouping the data into 3 clusters
k = 3
kmeans = KMeans(n_clusters=k, random_state=0).fit(X_proj)



# Plot the Original Classifications
plt.subplot(1, 2, 1)
plt.scatter(X_proj[:, 0], X_proj[:, 1], c=colormap[y], s=40)
plt.title('Real Classification')

# Plot the Models Classifications
plt.subplot(1, 2, 2)
plt.scatter(X_proj[:, 0], X_proj[:, 1], c=colormap[kmeans.labels_], s=40)
plt.title('K Mean Classification')

plt.show()