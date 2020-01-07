import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
from sklearn.model_selection import train_test_split
import glob
import os
import mahotas
import h5py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from os import listdir
from os.path import isfile, join, isdir
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pickle

bins = 8

#feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# # feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

featuresTest = []
labelsTest = []
listanh = []

#duong dan den file luu anh test
test_path = 'testanh'

# duong dan den file test trong dataset de lay ten các thư mục
# test2_path = 'test'


# my_path2_list = [f for f in listdir(test2_path) if isdir(test2_path)]
# test_labels = os.listdir(test2_path)
# test_labels.sort()
# pickle.dump((test_labels), open("listlabels", 'wb')) 

test_labels = pickle.load(open("listlabels", "rb"))

my_path_list = [f for f in listdir(test_path) if isdir(test_path)]
for my_path in my_path_list:
    onlyfiles = [f for f in listdir(join(test_path, my_path)) if isfile(join(test_path, my_path, f))]
    print(my_path)
    for file in onlyfiles:
        current_label = my_path
        image = cv2.imread(join(test_path, my_path, file))
        listanh.append(image)
        fv_histogram  = fd_histogram(image)
        fv_hu_moments  = fd_hu_moments(image)
        fv_haralick = fd_haralick(image)

        featureTest = np.hstack([fv_histogram,fv_hu_moments,fv_haralick])
        # Cập nhật featureTrain và labelsTrain
        featuresTest.append(featureTest)
        labelsTest.append(current_label)


        
print('[Test feature]: Finish!')

# Mã hóa dữ liệu labels
le          = LabelEncoder()
targerTest  = le.fit_transform(labelsTest)
print('[Encoding]: Finish!')


# load file scaler
# with open('scaler.pickle', 'rb') as f:
#     scaler = pickle.load(f)


# test thử khi chưa có file scaler ( neu có file scaler rồi thì xóa dong ke tiep)
scaler  = MinMaxScaler(feature_range=(0, 1))






rescaled_featuresT = scaler.fit_transform(featuresTest)
print('[Scale]: Finish!')




a = np.array(rescaled_featuresT)
#print(a.shape)

from joblib import dump, load

# load model da training len
clf = load('3svmrbf.joblib') 

x = a[8].reshape(1,-1)

u = clf.predict(x)[0]
print(test_labels[u])

size = (500,500)
for i in range(0,22):
	x = a[i].reshape(1,-1)
	u = clf.predict(x)[0]
	print(test_labels[u])
	img = cv2.resize(listanh[i], size)
	cv2.putText(img,"True: " + labelsTest[i], (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1.25 , (255,0,0), 4)
	cv2.putText(img,"Pred: " + test_labels[u], (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.25 , (255,0,0), 4)
	plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	plt.show()




