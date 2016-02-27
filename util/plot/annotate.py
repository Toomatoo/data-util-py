'''
Annotate every node in the picture.
'''

'''
data_proj is a 2-d array
'''
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(12,18))
ax = fig.add_subplot(111)

A = [data_proj[i][0] for i in range(len(data_proj))]
B = [data_proj[i][1] for i in range(len(data_proj))]
plt.plot(A,B, 'o')


'''
annotate(tag text, node tuple, others)
'''
# ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
#            arrowprops=dict(facecolor='black', shrink=0.05),
#            )
nodes = zip(A, B)
for i in range(len(nodes)):                                                # <--
    ax.annotate(animals_name[i], xy=nodes[i]) # <--

#plt.show()
plt.savefig('foo.png')
