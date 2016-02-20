import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

'''
Draw contour for a X, Y (Z here), where X is feature vector, Y is weight for
classification.

This one is a special for perceptron algorithm, but the drawing logistics is
below.
'''
def draw_contour(c, w, data):
    delta = 0.5
    x1 = np.arange(0, 11, 0.5)
    x2 = np.arange(0, 11, 0.5)

    X1, X2 = np.meshgrid(x1, x2)
    #print X2
    Z = np.array([[predict_one(c, w, [1, x1[j], x2[i]]) for j in range(len(x2))] for i in range(len(x1))])
    #print Z
    # the label
    plt.figure()

    for d in data:
        # print d
        if d[-1] == -1:
            plt.plot([d[1]],[d[2]],'*', color="red")
        else:
            plt.plot([d[1]],[d[2]],'o', color="blue")

    CS = plt.contour(x1, x2, Z, [0])
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Decision boundary with downsampled voted perceptron algorithm.')
    plt.show()
