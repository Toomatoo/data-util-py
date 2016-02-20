'''
Quatratic Kenel function.
:param z, x: pair of data points to be compute
:return: kernel value
:rtype: double
'''
def quadratic_kernel(z, x):
    res = 1 + np.dot(z, x)
    res = res**2
    return res


'''
Kernelized Perceptron for one data based on current alpha

:param x: the data point to be classified
:param alpha: current stage of alpha
:param data: the whole training data set
:return: the classification value (positive - positive class, otherwise)
'''
def classify_quadratic(x, alpha, data):
    sum_a = 0
    for i in range(len(data)):
        sum_a += alpha[i] * data[i][-1] * quadratic_kernel(data[i][0:-1], x)
    return sum_a
