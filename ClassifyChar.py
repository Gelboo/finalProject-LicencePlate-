from Preprocessing_Char import *
import keras

# one Hot_Encoding
# print y_train
num_classes = 63
# print len(y_train)
# print y_train.max(),y_train.min()
# print y_test.max(),y_test.min()
# print num_classes
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# break trainset into trainset and validationset
#take first 80% as train and 20% as validation
uptill = int(len(x_train)*0.8)
(x_train,x_valid) = x_train[:uptill],x_train[uptill:]
(y_train,y_valid) = y_train[:uptill],y_train[uptill:]

print x_train.shape
print x_valid.shape
print y_train.shape
print y_valid.shape



#define the model
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
model2 = Sequential()
model2.add(Conv2D(filters = 16,kernel_size=2,padding='same',activation='relu',input_shape=(100,100,3)))

model2.add(MaxPooling2D(pool_size=2))
model2.add(Conv2D(filters=32,kernel_size=2,padding='same',activation='relu'))
model2.add(MaxPooling2D(pool_size=2))
model2.add(Conv2D(filters=64,kernel_size=2,padding='same',activation='relu'))
model2.add(MaxPooling2D(pool_size=2))
model2.add(Conv2D(filters=128,kernel_size=2,padding='same',activation='relu'))
model2.add(MaxPooling2D(pool_size=2))

model2.add(Dropout(0.3))
model2.add(Flatten())
model2.add(Dense(500,activation='relu'))
model2.add(Dropout(0.3))

model2.add(Dense(63,activation='softmax'))
model2.summary()

#compile the model
model2.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint

# train the model
checkpointer = ModelCheckpoint(filepath='modelChar.weights.best.hdf5', verbose=1,
                               save_best_only=True)
# hist = model2.fit(x_train,y_train,batch_size=32,epochs=100,validation_data=(x_valid,y_valid),callbacks=[checkpointer],verbose=2,shuffle=True)
# load weight with best validation score
model2.load_weights('modelChar.weights.best.hdf5')

# calculate Cassification Accuracy
score = model2.evaluate(x_test,y_test,verbose=0)
print 'test accuracy',score[1]

Labels = Upper_Characters + lower_Characters + numbers
print Labels
im = cv2.imread('N.png')#'Pictures/Train_Images/TrainChar/A/1.png')
img = np.array([im])
# print img.shape
img2 = np.array([cv2.resize(im,(100,100))])
print Labels[np.argmax(model2.predict(img2))]

print x_test.shape
y_hat = (model2.predict(x_test))
# print y_hat
for i in range(x_test.shape[0]):
    plt.subplot(8,8,i+1,xticks=[],yticks=[])
    plt.imshow(np.squeeze(x_test[i][:,:,[2,1,0]]))
    pred_idx = np.argmax(y_hat[i])
    true_idx = np.argmax(y_test[i])
    plt.title("{} ({})".format(Labels[pred_idx],Labels[true_idx] ),color=("green" if pred_idx == true_idx else "red"))
plt.show()

from pyramid import *
import time
(winW,winH) = (100,100)

# img = cv2.imread('imgg.png')
# for resized in pyramid(img,scale = 1.5):
#     for (x,y,window) in sliding_window(resized,stepSize=32,windowSize=(winW,winH)):
#         #if window doesn't meet window size ignore it
#         if window.shape[0] != winH or window.shape[1] != winW:
#             continue
#         img = np.array([cv2.resize(window,(100,100))])
#         print Labels[np.argmax(model2.predict(img))]
#
#         clone = resized.copy()
#         cv2.rectangle(clone,(x,y),(x+winW,y+winH),(0,0,255),2)
#         cv2.imshow("window",clone)
#         cv2.waitKey(1)
#         time.sleep(0.25)
