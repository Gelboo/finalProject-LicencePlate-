import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

im = cv2.imread("imgg.png")
# # Convert to grayscale and apply Gaussian filtering
# im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
#
# # Threshold the image
# ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
#
# # Find contours in the image
# ctrs = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print len(ctrs)
#
# x,y,w,h = cv2.boundingRect(ctrs[3])
# cv2.rectangle(im, (x, y), (x+w,y+ h), (0, 255, 0), 3)
im3 = im.copy()

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

#################      Now finding Contours         ###################

_,ctrs,_ = cv2.findContours(thresh, 1, 2)
for cnt in ctrs:
    if cv2.contourArea(cnt)>500:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if h > 70:
            cv2.rectangle(im, (x, y), (x + w,y + h), (0, 255, 0), 3)
            roi = im[y:y+h,x:x+w]
            cv2.imshow("roi",roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    # Make the rectangular region around the digi
print len(ctrs)
cv2.imshow("image",im)
cv2.waitKey(0)
cv2.destroyAllWindows()
