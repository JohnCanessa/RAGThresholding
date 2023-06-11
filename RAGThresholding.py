# **** Imports ****
from matplotlib import pyplot as plt

from skimage import data, io, segmentation, color

#from skimage.future import graph
import skimage.graph as graph


# **** Read image ****
nature = io.imread('./images/pexels-nature.jpg')

# **** Display image ****
plt.figure(figsize=(8, 8))
plt.imshow(nature)
plt.title('Nature')
plt.show()


# **** Segments regions using k-means clustering
#      Balances color and space proximity -
#      high values emphasize spacial closeness
#      and the segments are squarish ****
labels1 = segmentation.slic(nature, 
                            compactness=35,
                            n_segments=500)

# **** Print labels1.shape ****
print(f'labels1.shape: {labels1.shape}')

# **** Print labels1 ****
print(f'labels1:\n{labels1}')

# **** segmented_overlay ****
segmented_overlay = color.label2rgb(labels1,
                                    nature,
                                    kind='overlay')

# **** Display segmented_overlay ****
plt.figure(figsize=(8, 8))
plt.imshow(segmented_overlay)
plt.title('Segmented Overlay')
plt.show()


# **** segmented_avg ****
segmented_avg = color.label2rgb(labels1,
                                nature,
                                kind='avg')

# **** Display segmented_avg ****
plt.figure(figsize=(8, 8))
plt.imshow(segmented_avg)
plt.title('Segmented Average')
plt.show()


# **** RAG thresholding merges segments of an image
#      based on how similar or dissimilar they are -
#      edbes are the difference in the mean color ****
g = graph.rag_mean_color(   nature,
                            labels1)

# **** labels2 - Combine regions separated by a 
#      weight less than threshold ****
labels2 = graph.cut_threshold(  labels1,
                                g,
                                thresh=35)

segmented_rag = color.label2rgb(labels2,
                                nature,
                                kind='avg')

# **** Display segmented_rag ****
plt.figure(figsize=(8, 8))
plt.imshow(segmented_rag)
plt.title('Segmented RAG - Threshold = 35')
plt.show()


# **** RAG thresholding merges segments of an image
#      based on how similar or dissimilar they are -
#      edbes are the difference in the mean color ****
g = graph.rag_mean_color(   nature,
                            labels1)

# **** labels2 - Combine regions separated by a 
#      weight less than threshold ****
labels2 = graph.cut_threshold(  labels1,
                                g,
                                thresh=15)

segmented_rag = color.label2rgb(labels2,
                                nature,
                                kind='avg')

# **** Display segmented_rag ****
plt.figure(figsize=(8, 8))
plt.imshow(segmented_rag)
plt.title('Segmented RAG - Threshold = 15')
plt.show()
