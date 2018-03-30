import cv2
import numpy

img  = cv2.imread('Pictures/English_license/Image_1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
Blur = cv2.blur(gray,(10,10))

_,thresh = cv2.threshold(Blur,80,255,cv2.THRESH_BINARY)

threshCopy = thresh.copy()

new_img ,contours,hierarchy = cv2.findContours(threshCopy,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,contours,3,(0,255,0),3)

cv2.imshow("img",thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
