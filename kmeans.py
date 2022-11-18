from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

plt.scatter(X[:,0], X[:,1],c=y)
plt.xlabel('Spea1 Length', fontsize=18)
plt.ylabel('Sepal Width', fontsize=18)

km = KMeans(n_clusters = 3)
km.fit(X)

centers = km.cluster_centers_
print(centers)

new_labels = km.labels_
print(new_labels)
print(y)

iris=datasets.load_iris()
x=iris.data[:,:2]
y=iris.target

fig, axes = plt.subplots(1,2 )
axes[0].scatter(X[:,0], X[:,1],c=y)
axes[1].scatter(X[:,0],x[:,1],c=y)
plt.scatter(x[:,0],x[:,1],c=new_labels)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='prism',edgecolor='k', s=75)
# plt.scatter(X[:, 0], X[:, 1], c=new_labels, cmap='jet',edgecolor='k', s=75)
axes[0].set_xlabel('Sepal length', fontsize=12)
axes[0].set_ylabel('Sepal width', fontsize=12)
axes[1].set_xlabel('Sepal length', fontsize=12)
axes[1].set_ylabel('Sepal width', fontsize=12)
axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=15)
axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=15)
axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('Predicted', fontsize=18)
