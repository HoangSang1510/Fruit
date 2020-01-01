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
from sklearn.externals import joblib
import pickle


# duong dan đen noi luu file  
h5_dataTrain   = 'output/outputtrain/datatrain.h5'
h5_labelsTrain = 'output/outputtrain/labelstrain.h5'
h5_dataTest    = 'output/outputtest/datatest.h5'
h5_labelsTest  = 'output/outputtest/labelstest.h5'


# Hàm trích xuất đặc trưng: Hu_moments, Haralick, Histogram
bins = 8
# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
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

# Tạo mảng rỗng lưu feature và label của 2 file train & test
featuresTrain = []
labelsTrain   = []
featuresTest  = []
labelsTest    = []

# Lấy feature & label của image trong train
train_path = 'dataset/train'
my_path_list = [f for f in listdir(train_path) if isdir(train_path)]
train_labels = os.listdir(train_path)
train_labels.sort()

for my_path in my_path_list:
    onlyfiles = [f for f in listdir(join(train_path, my_path)) if isfile(join(train_path, my_path, f))]
    for file in onlyfiles:
        current_label = my_path
        image = cv2.imread(join(train_path, my_path, file))
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)
        featureTrain = np.hstack([fv_histogram, fv_hu_moments, fv_haralick])
        # Cập nhật featureTrain và labelsTrain
        featuresTrain.append(featureTrain)
        labelsTrain.append(current_label)

    print('[Train feature]: {}'.format(current_label))
print('[Train feature]: Finish!')

# Lấy feature & label của image trong test
test_path = 'dataset/test'
my_path_list = [f for f in listdir(test_path) if isdir(test_path)]
test_labels = os.listdir(test_path)
test_labels.sort()

for my_path in my_path_list:
    onlyfiles = [f for f in listdir(join(test_path, my_path)) if isfile(join(test_path, my_path, f))]
    for file in onlyfiles:
        current_label = my_path
        image = cv2.imread(join(test_path, my_path, file))
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)
        featureTest = np.hstack([fv_histogram, fv_hu_moments, fv_haralick])
        # Cập nhật featureTrain và labelsTrain
        featuresTest.append(featureTest)
        labelsTest.append(current_label)
    print('[Test feature]: {}'.format(current_label))
print('[Test feature]: Finish!')

# Mã hóa dữ liệu labels
le          = LabelEncoder()
targetTrain = le.fit_transform(labelsTrain)
targerTest  = le.fit_transform(labelsTest)
print('[Encoding]: Finish!')

# scale dư lieu ve khoang 0 - 1 trươc khi lưu 
scaler  = MinMaxScaler(feature_range=(0, 1))
rescaled_featuresTr = scaler.fit_transform(featuresTrain)
rescaled_featuresT = scaler.fit_transform(featuresTest)

# lưu lại scaler
with open('scaler', 'wb') as f3:
    pickle.dump(scaler, f3)

print('[Scale]: Finish!')


# Lưu lại feature & label của train set vào file H5
h5f_dataTrain = h5py.File(h5_dataTrain, 'w')
h5f_dataTrain.create_dataset('dataset_1', data=np.array(featuresTrain))

h5f_labelTrain = h5py.File(h5_labelsTrain, 'w')
h5f_labelTrain.create_dataset('dataset_1', data=np.array(targetTrain))

h5f_dataTrain.close()
h5f_labelTrain.close()

# Lưu lại feature & label của test set vào file H5
h5f_dataTest = h5py.File(h5_dataTest, 'w')
h5f_dataTest.create_dataset('dataset_1', data=np.array(featuresTest))

h5f_labelTest = h5py.File(h5_labelsTest, 'w')
h5f_labelTest.create_dataset('dataset_1', data=np.array(targerTest))

h5f_dataTest.close()
h5f_labelTest.close()

print('[SAVING] Finish!')

