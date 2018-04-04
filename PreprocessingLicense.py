import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import glob

licenselist = glob.glob('Pictures/Train_Images/plate_images/*')

licensee = np.array([np.array(cv2.resize(cv2.imread(licensename),(100,100))) for licensename in licenselist])

# add Label for license y = 1
x_y = np.array([[img,1] for img in licensee])
print x_y.shape

# show first 36 licesne Examples
for i in range(36):
    plt.subplot(6,6,i+1,xticks=[],yticks=[])
    plt.imshow(licensee[i][:,:,[2,1,0]] , interpolation='nearest' , aspect='auto')
plt.show()

# load images other than license plate == 0
otherList = glob.glob('Pictures/Train_Images/other_images/*')
others = np.array([np.array(cv2.resize(cv2.imread(other),(100,100))) for other in otherList])


x_y = np.vstack((x_y,np.array([[img,0] for img in others])))
print x_y.shape

for i in range(36):
    plt.subplot(6,6,i+1,xticks=[],yticks=[])
    plt.imshow(others[i][:,:,[2,1,0]],interpolation='nearest',aspect='auto')
plt.show()

# randomize the image for no pattern
np.random.shuffle(x_y)
# show randomize picture
for i in range(36):
    plt.subplot(6,6,i+1,xticks=[],yticks=[])
    plt.imshow(x_y[i][0][:,:,[2,1,0]],interpolation='nearest',aspect='auto')
plt.show()

# take last 80% as train set
uptill = int(len(x_y)*0.8)

x_y_train = x_y[:uptill,:]
x_y_test = x_y[uptill:,:]

print x_y_train.shape
print x_y_test.shape

def divide_img_lbl(data):
    """ split data into image and label"""
    x = []
    y = []
    for [item,lbl] in data:
        x.append(item)
        y.append(lbl)
    x = np.array(x)
    y = np.array(y)
    return x,y

x_train,y_train = divide_img_lbl(x_y_train)

print x_train.shape
print y_train.shape

x_test,y_test = divide_img_lbl(x_y_test)

print x_test.shape
print y_test.shape

# rescale [0,255]  --> [0,1]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
# print x_train[0]

# break trainset into trainset and validationset
#take first 80% as train and 20% as validation
uptill = int(len(x_train)*0.8)
(x_train,x_valid) = x_train[:uptill],x_train[uptill:]
(y_train,y_valid) = y_train[:uptill],y_train[uptill:]

# print x_train.shape
# print x_valid.shape
# print y_train.shape
# print y_valid.shape
