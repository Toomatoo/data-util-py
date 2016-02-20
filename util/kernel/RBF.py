'''
RBF Kenel function.
:param z, x: pair of data points to be compute
:param sigma: sigma in the RNF kernel formula
:return: kernel value
:rtype: double
'''
def RBF_kernel(z, x, sigma):
    up = np.dot(np.subtract(x,z), np.subtract(x,z)) / (2 * sigma**2)
    return np.e ** (-up)

'''
Kernelized Perceptron for one data based on current alpha

:param x: the data point to be classified
:param alpha: current stage of alpha
:param data: the whole training data set
:return: the classification value (positive - positive class, otherwise)
'''
def classify_RBF(x, alpha, data, sigma):
    sum_a = 0
    for i in range(len(data)):
        sum_a += alpha[i] * data[i][-1] * RBF_kernel(data[i][0:-1], x, sigma)
    return sum_a
