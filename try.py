import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

im = cv2.imread("img2")

plt.imshow(im)
plt.show()

im_part = im[0:20,500:800]
plt.imshow(im_part)
plt.show()
