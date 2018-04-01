import time
import cv2
import numpy as np

# Method 1:
def pyramid(image,scale=1.5,minSize=(100,100)):
    yield image
    while True:
        # compute new size
        w = int(image.shape[1]/scale)
        image = cv2.resize(image,(w,w))

        # stop when reach the last layer based on minSize
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image
def sliding_window(image,stepSize,windowSize):
    #sliding the window across the image
    for y in range(0,image.shape[0],stepSize):
        for x in range(0,image.shape[1],stepSize):
            yield (x,y,image[y:y+windowSize[1],x:x+windowSize[0]])


# for (i,resized) in enumerate(pyramid(image,scale=1.5)):
#     cv2.imshow("Layer {}".format(i+1),resized)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# cv2.destroyAllWindows()

# Method 2: using Scikit-image
# from skimage.transform import pyramid_gaussian
# for (i,resized) in enumerate(pyramid_gaussian(image,downscale=2)):
#     # if the image is to small break the loop
#     if resized.shape[0] < 100 or resized.shape[1] < 100:
#         break
#     cv2.imshow("Layer {}".format(i+1),resized)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
