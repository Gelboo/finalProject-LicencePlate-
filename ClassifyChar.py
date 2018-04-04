from PreprocessingChar import *
import keras


# define the model
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
model3 = Sequential()
model3.add(Conv2D(filters = 16,kernel_size=2,padding='same',activation='relu',input_shape=(100,100,3)))

model3.add(MaxPooling2D(pool_size=2))
model3.add(Conv2D(filters=32,kernel_size=2,padding='same',activation='relu'))
model3.add(MaxPooling2D(pool_size=2))
model3.add(Conv2D(filters=64,kernel_size=2,padding='same',activation='relu'))
model3.add(MaxPooling2D(pool_size=2))
model3.add(Dropout(0.3))
model3.add(Flatten())
model3.add(Dense(500,activation='relu'))
model3.add(Dropout(0.4))

model3.add(Dense(1,activation='sigmoid'))
model3.summary()

# compile the model
model3.compile(loss='mean_squared_error',optimizer='rmsprop',metrics=['accuracy'])

# train the Model
from keras.callbacks import ModelCheckpoint
#train the model
checkpointer = ModelCheckpoint(filepath='model3.weights.best.hdf5',verbose=1,save_best_only=True)
hist = model3.fit(x_train,y_train,batch_size=32,epochs=100,validation_data=(x_valid,y_valid),callbacks=[checkpointer],verbose=2,shuffle=True)

# load weight with best validation score
model3.load_weights('model.weights.best.hdf5')

# calculate Cassification Accuracy
score = model3.evaluate(x_test,y_test,verbose=0)
print 'test accuracy',score[1]

Labels = ['no Char','Char']
y_hat = model3.predict(x_test)

y_hat = np.round(y_hat).astype('int').flatten()
# print y_hat
for i in range(20):
    plt.subplot(4,5,i+1,xticks=[],yticks=[])
    plt.imshow(np.squeeze(x_test[i][:,:,[2,1,0]]))
    # print y_hat[i],y_test[i]
    plt.title("{} ({})".format(Labels[y_hat[i]],Labels[y_test[i][0]] ),color=("green" if y_hat[i] == y_test[i] else "red"))
plt.show()
