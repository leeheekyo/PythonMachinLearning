from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[4, 2], [1, 4], [1, 0],
    [9, 2], [6, 4], [6, 0]])
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(x)
print(kmeans.labels_)
kmeans.predict([[0, 0], [12, 3]])
print(kmeans.cluster_centers_)

# fig, ax = plt.subplots(figsize=(8, 6))
# plt.scatter(X[:,0], X[:,1],  marker = 'o', 
#             c=kmeans.cluster_centers_)
# plt.scatter(X[:,0], X[:,1],  
#             marker = 's', s=200, c=[0, 1, 2])

# x0=np.array([])
# x1=np.array([])
# idx=0
# for item in kmeans.labels_:
#     if item == True:
#         x0 = np.append(x0,x[idx])
#         # x0.append(x[idx])
#     else:
#         x1 = np.append(x1,x[idx])
#         # x1.append(x[idx])
#     idx+=1

# print(x0)
# print(x1)

plt.scatter(x[:,0], x[:,1], c=kmeans.labels_, marker = 'o')
plt.scatter(x[:,0], x[:,1], c=kmeans.labels_, marker = 'x')

plt.show()
