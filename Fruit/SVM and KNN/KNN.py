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

#duong dan den cac file trich xuat dac trung da luu
h5_dataTrain   = 'output/outputtrain/datatrain.h5'
h5_labelsTrain = 'output/outputtrain/labelstrain.h5'
h5_dataTest    = 'output/outputtest/datatest.h5'
h5_labelsTest  = 'output/outputtest/labelstest.h5'

# Đọc dữ liệu từ file H5
h5f_dataTrain  = h5py.File(h5_dataTrain, 'r')
h5f_labelTrain = h5py.File(h5_labelsTrain, 'r')

global_features_string1 = h5f_dataTrain['dataset_1']
global_labels_string1   = h5f_labelTrain['dataset_1']

training_features = np.array(global_features_string1)
training_labels   = np.array(global_labels_string1)

h5f_dataTrain.close()
h5f_labelTrain.close()

h5f_dataTest  = h5py.File(h5_dataTest, 'r')
h5f_labelTest = h5py.File(h5_labelsTest, 'r')

global_features_string2 = h5f_dataTest['dataset_1']
global_labels_string2   = h5f_labelTest['dataset_1']

test_features = np.array(global_features_string2)
test_labels   = np.array(global_labels_string2)

h5f_dataTest.close()
h5f_labelTest.close()
print('[READING] Finish!')

from joblib import dump, load


# Mô hình KNN


print('Begin Training with KNN ....')
knn_clf = KNeighborsClassifier(n_jobs= - 1, weights='distance', n_neighbors= 11) 
knn_clf.fit(training_features, training_labels)
y_knn_pred = knn_clf.predict(test_features)
print('End of Training....')
print('KNN accuracy score: ',accuracy_score(test_labels, y_knn_pred)*100,'%')

dump(knn_clf, 'KNN.joblib')