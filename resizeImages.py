import cv2
import numpy as np
import glob

path = "Pictures/carTires"

imageList = glob.glob(path+'/*')
# print imageLsist

i = 0
for image in imageList:
    im = cv2.imread(image)
    im = cv2.resize(im,(100,100))
    i += 1
    name = path+"/resize_tires_" + str(i) +".png"
    cv2.imwrite(name,im)
