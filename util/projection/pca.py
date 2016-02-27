'''
PCA projection

:param data: data to estimate the variable X
:param k: projection dimension
'''
import numpy as np
import operator


'''
Input:  original data and dimention number k,
Output: new data in the generated new k-dimensional space.

:param data: orignal list of data points
:param k: the dimension of new space


1. Compute new space with PCA theory:
    1.1 Compute covariance matrix: using maximum likelihood method
    1.2 Compute eigenvalues and eigenvectors for the matrix, and sort
    1.3 Pick out the largest k eigenvalues.
2. Project every data into the new space.
'''
def pca(data, k):
    space = get_space(data, k)
    #print space
    data_proj = []
    for d in data:
        new_d = []
        for i in range(k):
            new_d.append(np.dot(d, space[i].T))
        data_proj.append(new_d)
    return data_proj


def ML_cov(data):
    u = np.average(data, axis=0)
    #print u
    expec = np.zeros((len(data[0]), len(data[0])))
    for d in data:
        expec += np.dot((d - u).T, d-u)
    return expec / len(data)

def get_space(data, k):
    cov = ML_cov(data)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    # sort
    eigen = {}
    for i in range(len(eigenvalues)):
        eigen[eigenvalues[i]] = eigenvectors[i]
    eigen = sorted(eigen.items(), key=operator.itemgetter(0), reverse=True)
    res = []
    for i in range(k):
        res.append(eigen[i][1])
    return np.array(res)


data_proj = pca(data, 2)
