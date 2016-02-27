from scipy.cluster.hierarchy import dendrogram, linkage, average
from matplotlib import pyplot as plt

'''
Call hierarchy clustering with average linkage.

:param: data: list of data points
:param: str: name of distance function
'''
Z = linkage(data, 'average')


'''
Draw a picture of hierarchy clustering
Call `dendrogram`
'''
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.ylabel('Animal Name')
plt.xlabel('distance')

dendrogram(
    Z,
    leaf_font_size=8,  # font size for the x axis labels
    orientation = 'right',
    labels = animals_name
)
plt.show()
