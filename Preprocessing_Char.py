import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

Upper_Characters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R',
'S','T','U','V','W','X','Y','Z']
lower_Characters = [c.lower() for c in Upper_Characters]
numbers = ['0','1','2','3','4','5','6','7','8','9','10']
def prepare_X_Y(path):
    lbl = 0
    charList = glob.glob(path+'A/*')
    chars = np.array([np.array(cv2.resize(cv2.imread(char),(100,100))) for char in charList])
    x_y = np.array([[img,lbl] for img in chars])
    lbl += 1
    digits = Upper_Characters[1:] + lower_Characters + numbers
    for d in digits:
        # print d
        charList = glob.glob(path + d + '/*')
        chars = np.array([np.array(cv2.resize(cv2.imread(char),(100,100))) for char in charList])
        x_y = np.vstack((x_y,np.array([[img,lbl] for img in chars])))
        lbl += 1
    print x_y.shape
    # print x_y[:,1]
    np.random.shuffle(x_y)
    return x_y
x_y_train = prepare_X_Y('Pictures/Train_Images/TrainChar/')
x_y_test = prepare_X_Y('Pictures/Test_images/Test_char/')
for i in range(36):
    plt.subplot(6,6,i+1,xticks=[],yticks=[])
    plt.imshow(x_y_train[i,0][:,:,[2,1,0]],interpolation='nearest',aspect='auto')
plt.show()




x_y_train = x_y_train
x_y_test = x_y_test

print x_y_train.shape
print x_y_test.shape

def divide_img_lbl(data):
    """ split data into image and label"""
    x = []
    y = []
    for [item,lbl] in data:
        x.append(item)
        y.append([lbl])
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
