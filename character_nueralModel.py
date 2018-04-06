from Preprocessing_DetectChar import *
import keras





#define the model
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
model2 = Sequential()
model2.add(Flatten(input_shape=x_train.shape[1:]))
model2.add(Dense(1000,activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(500,activation='relu'))
model2.add(Dropout(0.3))

model2.add(Dense(36,activation='softmax'))
model2.summary()

#compile the model
model2.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint

# train the model
checkpointer = ModelCheckpoint(filepath='neuralmodelChar.weights.best.hdf5', verbose=1,
                               save_best_only=True)
hist = model2.fit(x_train,y_train,batch_size=32,epochs=100,validation_data=(x_valid,y_valid),callbacks=[checkpointer],verbose=2,shuffle=True)
# load weight with best validation score
model2.load_weights('neuralmodelChar.weights.best.hdf5')

# calculate Cassification Accuracy
score = model2.evaluate(x_test,y_test,verbose=0)
print 'test accuracy',score[1]

Labels = Upper_Characters + numbers
y_hat = (model2.predict(x_test))
# print y_hat
for i in range(x_test.shape[0]):
    plt.subplot(8,8,i+1,xticks=[],yticks=[])
    plt.imshow(np.squeeze(x_test[i][:,:,[2,1,0]]))
    pred_idx = np.argmax(y_hat[i])
    true_idx = np.argmax(y_test[i])
    plt.title("{} ({})".format(Labels[pred_idx],Labels[true_idx] ),color=("green" if pred_idx == true_idx else "red"))
plt.show()
