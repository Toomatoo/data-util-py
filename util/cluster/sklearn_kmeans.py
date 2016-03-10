from sklearn.cluster import KMeans

'''
Call the kmeans function is sklearn

:param data: list of data points, n*p
:param K: k clusters
'''
def call_kmeans(data, K):
    kmeans = KMeans(n_clusters=K, precompute_distances=True, n_jobs=-1)
    return kmeans.fit_predict(data)
    
