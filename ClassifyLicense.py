from PreprocessingLicense import *
import keras


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
model.summary()

# compile the model
model.compile(loss='mean_squared_error',optimizer='rmsprop',metrics=['accuracy'])

# train the Model
from keras.callbacks import ModelCheckpoint
#train the model
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5',verbose=1,save_best_only=True)
# hist = model.fit(x_train,y_train,batch_size=32,epochs=100,validation_data=(x_valid,y_valid),callbacks=[checkpointer],verbose=2,shuffle=True)

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

im = cv2.imread('image6.jpg')
img = np.array([im])
# print img.shape
img2 = np.array([cv2.resize(im,(100,100))])
# print model.predict(img2)


(winW,winH) = (130,80)
stepSizeH,stepSizeV = 50,winH

from pyramid import *
import time
i = 0
for resized in pyramid(im,scale = 3,minSize=(200,200)):
    for (x,y,window) in sliding_window(resized,stepSizeH=stepSizeH,stepSizeV=stepSizeV,windowSize=(winW,winH)):
        #if window doesn't meet window size ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        img = np.array([cv2.resize(window,(100,100))])
        prediction = model.predict(img)
        print prediction
        color = (0,0,255)
        if prediction == 1:
            cv2.imwrite("PossibleLicense/img"+str(i)+".png",cv2.resize(window,(400,400)))
            i+=1
            color = (255,0,0)
        clone = resized.copy()
        cv2.rectangle(clone,(x,y),(x+winW,y+winH),color,2)
        cv2.imshow("window",clone)
        cv2.waitKey(1)
        time.sleep(0.5)
