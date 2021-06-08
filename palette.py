"""
https://medium.com/towardssingularity/k-means-clustering-for-image-segmentation-using-opencv-in-python-17178ce3d6f3#:~:
text=CV2.KMEANS%20Return%20Value%20compactness%20%3A%20It%20is%20the,
%3A%20This%20is%20array%20of%20centers%20of%20clusters
"""
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

df = pd.read_csv('colors.csv')
df.columns = ['name', 'formatted_name', 'hex', 'r', 'g', 'b']


def min_color(r, g, b):
    """
    :param r: R
    :param g: G
    :param b: B
    :return: closest color from dataset
    """
    error = np.infty
    idx = 0
    for col in range(df.shape[0]):
        red = df.loc[col, 'r']
        gr = df.loc[col, 'g']
        blue = df.loc[col, 'b']
        r_e = (red - r) ** 2
        g_e = (gr - g) ** 2
        b_e = (blue - b) ** 2
        total_e = r_e + g_e + b_e
        if total_e < error:
            idx = col
            error = total_e
    return df.loc[idx, 'formatted_name'], df.loc[idx, 'r'], df.loc[idx, 'g'], df.loc[idx, 'b']


# Load img
image = cv2.imread('suz.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img = image.reshape((-1, 3))
img = np.float32(img)

# max iter 100, accuracy 0.9
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.9)
k = 7
ret, labels, centers = cv2.kmeans(img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

palette = []
color_names = []
for i in centers:
    info = min_color(i[0], i[1], i[2])
    palette.append(Line2D([0], [0], color=(info[1] / 255, info[2] / 255, info[3] / 255), lw=4))
    color_names.append(info[0])

# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
# reshape back into original image shape
segmented_image = segmented_data.reshape(image.shape)
fig, ax = plt.subplots()
ax.legend(palette, color_names)
plt.imshow(segmented_image)
plt.show()



# cv2.imwrite('test.jpeg', segmented_image)
