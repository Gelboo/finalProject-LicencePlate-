from pyramid import *
import time
import cv2
image = cv2.imread('Image_37.jpg')
(winW,winH) = (50,100)
i = 0
for resized in pyramid(image,scale = 1.5):
    for (x,y,window) in sliding_window(resized,stepSize=15,windowSize=(winW,winH)):
        #if window doesn't meet window size ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # cv2.imwrite("Pictures/CarParts/img"+str(i)+".png",window)
        # i += 1
        clone = resized.copy()
        cv2.rectangle(clone,(x,y),(x+winW,y+winH),(0,0,255),2)
        cv2.imshow("window",clone)
        cv2.waitKey(1)
        time.sleep(0.25)
