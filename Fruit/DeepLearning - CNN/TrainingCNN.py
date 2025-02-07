from sklearn.datasets import load_files
import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
import cv2


# duong dan den thu muc dataset
train_dir = 'dataset/train'
test_dir = 'dataset/test'

def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    target_labels = np.array(data['target_names'])
    return files,targets,target_labels
    
x_train, y_train,target_labels = load_dataset(train_dir)
x_test, y_test,_ = load_dataset(test_dir)
print('Loading complete!')

print('Training set size : ' , x_train.shape[0])
print('Testing set size : ', x_test.shape[0])

no_of_classes = len(np.unique(y_train))
print(no_of_classes)



from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train,no_of_classes)
y_test = np_utils.to_categorical(y_test,no_of_classes)

x_test,x_valid = x_test[9000:],x_test[:9000]
y_test,y_vaild = y_test[9000:],y_test[:9000]
print('Vaildation X : ',x_valid.shape)
print('Vaildation y :',y_vaild.shape)
print('Test X : ',x_test.shape)
print('Test y : ',y_test.shape)

def convert_image_to_array(files):
    images_as_array=[]
    for file in files:
        # Convert to Numpy Array
        images_as_array.append(img_to_array(load_img(file)))
    return images_as_array


x_train = np.array(convert_image_to_array(x_train))
print('Training set shape : ',x_train.shape)

x_valid = np.array(convert_image_to_array(x_valid))
print('Validation set shape : ',x_valid.shape)

x_test = np.array(convert_image_to_array(x_test))
print('Test set shape : ',x_test.shape)

print('1st training image shape ',x_train[0].shape)


# sacle du lieu trong khoang tu 0 - 1
x_train = x_train.astype('float32')/255
x_valid = x_valid.astype('float32')/255
x_test = x_test.astype('float32')/255




#Hien 10 hinh len để xem 
# fig = plt.figure(figsize =(30,5))
# for i in range(10):
#     ax = fig.add_subplot(2,5,i+1,xticks=[],yticks=[])
#     ax.imshow(np.squeeze(x_train[i]))
#plt.show()


#Simple CNN from scratch - we are using 3 Conv layers followed by maxpooling layers.
# At the end we add dropout, flatten and some fully connected layers(Dense).
print('Building a model...')
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K

model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 2,input_shape=(100,100,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 32,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 64,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 128,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(120,activation = 'softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print('Compiled init Model!')
model.save('fruit_cnn.model')


print('Start to traing...')
batch_size = 32
checkpointer = ModelCheckpoint(filepath = 'cnn_from_scratch_fruits.hdf5', verbose = 1, save_best_only = True)
history = model.fit(x_train,y_train,
        batch_size = 32,
        epochs=30,
        validation_data=(x_valid, y_vaild),
        callbacks = [checkpointer],
        verbose=2, shuffle=True)

print('End of traing...')



# # load the weights that yielded the best validation accuracy
# model.load_weights('cnn_from_scratch_fruits.hdf5')
# score = model.evaluate(x_test, y_test, verbose=0)
# print('\n', 'Test accuracy:', score[1])

# # y_pred = model.predict(x_test)
# # # plot a random sample of test images, their predicted labels, and ground truth
# # fig = plt.figure(figsize=(16, 9))
# # for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16, replace=False)):
# #     ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
# #     ax.imshow(np.squeeze(x_test[idx]))
# #     pred_idx = np.argmax(y_pred[idx])
# #     true_idx = np.argmax(y_test[idx])
# #     ax.set_title("{} ({})".format(target_labels[pred_idx], target_labels[true_idx]),
# #                  color=("green" if pred_idx == true_idx else "red"))
# # plt.show()



