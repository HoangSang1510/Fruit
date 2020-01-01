import cv2
import tensorflow as tf
import h5py
import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from os import listdir
from os.path import isfile, join, isdir


# duong dan den thu muc train và thu muc test
train_dir = 'dataset/train'
test_dir = 'dataset/test'
labels_dir = 'dataset/test'

my_path_list = [f for f in listdir(labels_dir) if isdir(labels_dir)]

list_labels = []

for my_path in my_path_list:
    onlyfiles = [f for f in listdir(join(labels_dir, my_path)) if isfile(join(labels_dir, my_path, f))]
    name = my_path
    list_labels.append(name)


def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    target_labels = np.array(data['target_names'])
    return files,targets,target_labels
    
x_test, y_test,_ = load_dataset(labels_dir)

print('Load data test complete!')





from keras.utils import np_utils

y_test = np_utils.to_categorical(y_test,120)


x_test,x_valid = x_test[100:],x_test[:100]

y_test,y_vaild = y_test[100:],y_test[:100]


def convert_image_to_array(files):
    images_as_array=[]
    for file in files:
        # Convert to Numpy Array
        images_as_array.append(img_to_array(load_img(file)))
    return images_as_array



x_valid = np.array(convert_image_to_array(x_valid))
print('Validation set shape : ',x_valid.shape)

x_test = np.array(convert_image_to_array(x_test))
#print(x_test)
print('Test set shape : ',x_test.shape)



#time to re-scale so that all the pixel values lie within 0 to 1
x_valid = x_valid.astype('float32')/255
x_test = x_test.astype('float32')/255



model = tf.keras.models.load_model("cnnS2.h5")


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#model.summary()

print('Load CNN model complete!')


y_pred = model.predict(x_test)

#plot a random sample of test images, their predicted labels, and ground truth
fig = plt.figure(figsize=(16, 9))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16, replace=False)):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[idx]))
    pred_idx = np.argmax(y_pred[idx])
    true_idx = np.argmax(y_test[idx])
    ax.set_title("{} ({})".format(list_labels[pred_idx], list_labels[true_idx]),
                 color=("green" if pred_idx == true_idx else "red"))
plt.show()



# img = cv2.imread('bana.jpg')
# img = cv2.resize(img,(100,100))
# img = img.astype('float32')/255
# img = np.reshape(img,[1,100,100,3])

# classes = model.predict_classes(img)

# print(classes)

# print(list_labels[classes])




