from PreprocessingLicense import *
import keras

print x_train.shape[1:]
# define the model
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(500,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

# compile the model
model.compile(loss='mean_squared_error',optimizer='rmsprop',metrics=['accuracy'])

# train the Model
from keras.callbacks import ModelCheckpoint
#train the model
checkpointer = ModelCheckpoint(filepath='model_neural.weights.best.hdf5',verbose=1,save_best_only=True)
hist = model.fit(x_train,y_train,batch_size=32,epochs=100,validation_data=(x_valid,y_valid),callbacks=[checkpointer],verbose=2,shuffle=True)

# load weight with best validation score
model.load_weights('model_neural.weights.best.hdf5')

# calculate Cassification Accuracy
score = model.evaluate(x_test,y_test,verbose=0)
print 'test accuracy',score[1]

y_hat = model.predict(x_test)
# print y_test
# print y_hat
Labels = ['No License','License']



y_hat = np.round(y_hat).astype('int').flatten()
# print y_hat
for i in range(20):
    plt.subplot(4,5,i+1,xticks=[],yticks=[])
    plt.imshow(np.squeeze(x_test[i][:,:,[2,1,0]]))
    plt.title("{} ({})".format(Labels[y_hat[i]],Labels[y_test[i]] ),color=("green" if y_hat[i] == y_test[i] else "red"))
plt.show()
