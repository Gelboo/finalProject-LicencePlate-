from Preprocessing import *
import keras

# break trainset into trainset and validationset
#take first 80% as train and 20% as validation
uptill = int(len(x_train)*0.8)
(x_train,x_valid) = x_train[:uptill],x_train[uptill:]
(y_train,y_valid) = y_train[:uptill],y_train[uptill:]

# print x_train.shape
# print x_valid.shape
# print y_train.shape
# print y_valid.shape

# define the model
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
model = Sequential()
model.add(Conv2D(filters = 16,kernel_size=2,padding='same',activation='relu',input_shape=(100,100,3)))

model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(1,activation='sigmoid'))
# model.summary()

# compile the model
model.compile(loss='mean_squared_error',optimizer='rmsprop',metrics=['accuracy'])

# train the Model
from keras.callbacks import ModelCheckpoint
#train the model
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5',verbose=1,save_best_only=True)
hist = model.fit(x_train,y_train,batch_size=32,epochs=100,validation_data=(x_valid,y_valid),callbacks=[checkpointer],verbose=2,shuffle=True)

# load weight with best validation score
model.load_weights('model.weights.best.hdf5')

# calculate Cassification Accuracy
score = model.evaluate(x_test,y_test,verbose=0)
print 'test accuracy',score[1]

# test_img = []
# test_img.append(np.array(cv2.resize(cv2.imread('Pictures/Test_images/person.jpg'),(40,40))))
# test_img.append(np.array(cv2.resize(cv2.imread('Pictures/Test_images/Image_2.jpg'),(40,40))))
# test_img.append(np.array(cv2.resize(cv2.imread('Pictures/Test_images/ActiOn_3.jpg'),(40,40))))
# test_img.append(np.array(cv2.resize(cv2.imread('Pictures/Test_images/Image_86.jpg'),(40,40))))
# test_img.append(np.array(cv2.resize(cv2.imread('Pictures/Test_images/ActiOn_74.jpg'),(40,40))))
# test_img = np.array(test_img)
# y_test_img = [0,1,1,1,1]
#
# print test_img.shape
# y_hat = model.predict(x_test)
# print y_test
# print y_hat
Labels = ['No License','License']



# y_hat = np.round(y_hat).astype('int').flatten()
# # print y_hat
# for i in range(20):
#     plt.subplot(4,5,i+1,xticks=[],yticks=[])
#     plt.imshow(np.squeeze(x_test[i][:,:,[2,1,0]]))
#     plt.title("{} ({})".format(Labels[y_hat[i]],Labels[y_test[i]] ),color=("green" if y_hat[i] == y_test[i] else "red"))
# plt.show()

#
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.patches as patches

im = cv2.imread('Pictures/licenseWithPartOfCar/Image_3.jpg')
img = np.array([im])
# print img.shape
img2 = np.array([cv2.resize(im,(100,100))])
# print model.predict(img2)


(winW,winH) = (100,100)
from pyramid import *
import time

for resized in pyramid(im,scale = 1.5):
    for (x,y,window) in sliding_window(resized,stepSize=32,windowSize=(winW,winH)):
        #if window doesn't meet window size ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        img = np.array([cv2.resize(window,(100,100))])
        print model.predict(img)

        clone = resized.copy()
        cv2.rectangle(clone,(x,y),(x+winW,y+winH),(0,0,255),2)
        cv2.imshow("window",clone)
        cv2.waitKey(1)
        time.sleep(0.25)
