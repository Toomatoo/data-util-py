import numpy as np

'''
Read data from a text file.
Each line denotes a data point (x y), splitted by single space.

:parram filename
:type filename: str
:returns: list of data
:rtype: list of list
'''
def read_data(filename):
    # read out the training data
    # return x[] y[]
    data = []
    fr = open(filename, "r")
    for l in fr.readlines():
        strs = l.split()
        p = [1]
        for i in range(len(strs)):
            p.append((int)(strs[i]))
        data.append(p)
    return data

    
## data = read_data("data1.txt")


'''
Permute a list of data.

:parram data
:type data: list of list
:returns: list of data
:rtype: list of list
'''
def permute(data):
    return np.random.permutation(data)


'''
One round perceptron (go through data for once).
You can custermize below to modify the perceptron updata rule.

:parram c
:type c: list of double
:parram w
:type w: list of double
:returns: new c and w
:rtype: lists
'''
def one_perceptron(c, w, data):
    # Go through every point
    # if there is a misclassification
    #   create a new w and start to record the success time (c)
    # else
    #   increase the success time of the current w
    for d in data:
        if (np.dot(w[-1], d[0:-1]) * d[-1]) <= 0:
            w = np.append(w, [w[-1] + d[-1]*d[0:-1]], axis=0)
            c = np.append(c, [1], axis=0)
            # print w
        else:
            c[-1] += 1
    return c, w


'''
The whole Logistics for Perceptron.
:parram data
:type data: list
:parram T
:type T: number of iteration
:returns c, w, aver_w
'''
def gen_perceptron(data, T):
    # Initialize the parameters c, w
    # Go for T times of perceptron iteration
    # return a average w
    c = np.array([0])
    w = np.array([np.zeros(len(data[0])-1)])
    for t in range(T):
        data = permute(data)
        c, w = one_perceptron(c, w, data)

    aver_w = np.zeros(len(data[0])-1)
    for i in range(len(w)):
        aver_w += c[i] * w[i]
    return c, w, aver_w



## c, w, aver_w = gen_perceptron(data, 100)

'''
A simple classifcation rule.
You can modify it.
'''
def predict(data, w):
    count = 0
    for d in data:
        if (np.dot(w, d[0:-1]) * d[-1]) > 0:
            count += 1
    return float(count) / len(data)

## print predict(data, aver_w)
