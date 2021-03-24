from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np

X, Y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None)


kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001).fit(X)

predict = kmeans.predict(X)

c1 = list()
c2 = list()

for i in range(len(predict)):
    if predict[i] == 0:
        c1.append((X[i][0], X[i][1]))
    else:
        c2.append((X[i][0], X[i][1]))
        

c1 = np.array(c1)
c2 = np.array(c2)

kmeans_c1 = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001).fit(c1)
kmeans_c2 = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001).fit(c2)


predict_c1 = kmeans_c1.predict(c1)
predict_c2 = kmeans_c2.predict(c2)

print(predict_c2)

c3 = list()
c4 = list()

c5 = list()
c6 = list()

for i in range(len(predict_c1)):
    if predict_c1[i] == 0:
        c3.append((c1[i][0], c1[i][1]))
    else:
        c4.append((c1[i][0], c1[i][1]))

for i in range(len(predict_c2)):
    if predict_c2[i] == 0:
        c5.append((c2[i][0], c2[i][1]))
    else:
        c6.append((c2[i][0], c2[i][1]))

c3 = np.array(c3)
c4 = np.array(c4)

c5 = np.array(c5)
c6 = np.array(c6)





plt.scatter(c3[:,0], c3[:,1], color='b', marker='*')
plt.scatter(c4[:,0], c4[:,1], color='y', marker='^')
plt.scatter(c5[:,0], c5[:,1], color='r', marker='+')
plt.scatter(c6[:,0], c6[:,1], color='g', marker='.')



plt.show()
        

        



















