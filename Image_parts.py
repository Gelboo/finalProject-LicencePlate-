from pyramid import *
import time
import cv2
import matplotlib.pyplot as plt
image = cv2.imread('letters2.png')
(winW,winH) = (50,100)
plt.imshow(image)
plt.show()
i = 0
for resized in pyramid(image,scale = 1.5,minSize=(image.shape[1],image.shape[0])):
    for (x,y,window) in sliding_window(resized,stepSizeH=winW,stepSizeV=75,windowSize=(winW,winH)):
        #if window doesn't meet window size ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        cv2.imwrite("Pictures/ClassifyChar/img"+str(i)+".png",window)
        i += 1
        clone = resized.copy()
        cv2.rectangle(clone,(x,y),(x+winW,y+winH),(0,0,255),2)
        cv2.imshow("window",clone)
        cv2.waitKey(1)
        time.sleep(0.5)
