
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
import pandas as pd
from skimage.color import rgb2gray 
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.model_selection import train_test_split
from skimage import data, color, feature
from skimage.feature import hog
import glob
import os
import h5py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from os import listdir
from os.path import isfile, join, isdir
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import cv2
h5_dataTrain          = 'output/outputtrain/datatrainHOGscale.h5'
h5_labelsTrain        = 'output/outputtrain/labelstrainHOGscale.h5'

h5_dataTest          = 'output/outputtest/datatestHOGscale.h5'
h5_labelsTest        = 'output/outputtrain/labelstestHOGscale.h5'



# # lay feature + lable cua hinh trong train
train_path = "dataset/train"
my_path_list = [f for f in listdir(train_path) if isdir(train_path)]
train_labels = os.listdir(train_path)

label  = []
arr = []
for my_path in my_path_list:
	onlyfiles = [f for f in listdir(join(train_path, my_path)) if isfile(join(train_path, my_path, f))]
	for file in onlyfiles:
		current_label = my_path
		image = np.asarray(plt.imread(join(train_path, my_path, file)))
		arr.append(image)
		label.append(current_label)
print('Get data train: OK')

# lay feature + lable cua hinh trong test
test_path = "dataset/test"
my_path_list = [f for f in listdir(test_path) if isdir(test_path)]
test_labels = os.listdir(test_path)

label2 = []
arr2 = []
for my_path in my_path_list:
	onlyfiles = [f for f in listdir(join(test_path, my_path)) if isfile(join(test_path, my_path, f))]
	for file in onlyfiles:
		current_label = my_path
		image = np.asarray(plt.imread(join(test_path, my_path, file)))
		arr2.append(image)
		label2.append(current_label)
print('Get data test: OK')



def preprocessing1(arr):
    arr_prep=[]    
    for i in range(np.shape(arr)[0]):
        img=cv2.cvtColor(arr[i], cv2.COLOR_BGR2GRAY)
        img=resize(img, (100, 100),anti_aliasing=True)        
        arr_prep.append(img)    
    return arr_prep

# HOG 
def FtrExtractHOG(img):
    ftr,_=hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=False)   
    return ftr    

# fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
#                     cells_per_block=(1, 1), visualize=True, multichannel=True)


def featureExtraction1(arr):    
	arr_feature=[]    
	for i in range(np.shape(arr)[0]):
	    arr_feature.append(FtrExtractHOG(arr[i]))
	return arr_feature



#Preprocessing: Tiền xử lí
X_trainp=preprocessing1(arr) 
X_testp=preprocessing1(arr2) 
print('Preprocessing: OK')



#Feature Extraction: Tách đặc trưng 
X_trainftr=featureExtraction1(X_trainp) 
X_testftr=featureExtraction1(X_testp)
print('featureExtraction: OK')

#scale du lieu
X_trainftr = preprocessing.scale(X_trainftr) 
X_testftr = preprocessing.scale(X_testftr) 


#mã hóa label thanh số để lưu trữ
le          = LabelEncoder()
target      = le.fit_transform(label)
target2      = le.fit_transform(label2)


# #save lại freture + lable cua tap training
h5f_data1 = h5py.File(h5_dataTrain, 'w')
h5f_data1.create_dataset('dataset_1', data=np.array(X_trainftr))

h5f_label1 = h5py.File(h5_labelsTrain, 'w')
h5f_label1.create_dataset('dataset_1', data=np.array(target))

h5f_data1.close()
h5f_label1.close()

#save lại freture + lable cua tap test
h5f_data2 = h5py.File(h5_dataTest, 'w')
h5f_data2.create_dataset('dataset_1', data=np.array(X_testftr))

h5f_label2 = h5py.File(h5_labelsTest, 'w')
h5f_label2.create_dataset('dataset_1', data=np.array(target2))

h5f_data2.close()
h5f_label2.close()

print("[STATUS] end of SAVE")




#doc du lieu tu file.h5
print('Begin Reading...')
h5f_data1  = h5py.File(h5_dataTrain, 'r')
h5f_label1 = h5py.File(h5_labelsTrain, 'r')

global_features_string1 = h5f_data1['dataset_1']
global_labels_string1   = h5f_label1['dataset_1']

training_features = np.array(global_features_string1)
training_labels   = np.array(global_labels_string1)

h5f_data1.close()
h5f_label1.close()

h5f_data2  = h5py.File(h5_dataTest, 'r')
h5f_label2 = h5py.File(h5_labelsTest, 'r')

global_features_string2 = h5f_data2['dataset_1']
global_labels_string2   = h5f_label2['dataset_1']

test_features = np.array(global_features_string2)
test_labels   = np.array(global_labels_string2)

h5f_data1.close()
h5f_label1.close()

print("[STATUS] end of READ")



# Mô hình SVM
print('Begin Training with SVM ....')
svm_clf = SVC(gamma = 'auto', probability=True )
svm_clf.fit(training_features, training_labels)
y_svm_pred = svm_clf.predict(test_features)
print('End of Training....')
print('SVM accuracy score: ',accuracy_score(test_labels, y_svm_pred)*100,'%')


# from sklearn.externals import joblib
# filename = 'SVMmodel.sav'
# joblib.dump(svm_clf, filename)

# loaded_model = joblib.load(filename)
# result = loaded_model.score(test_features, test_labels)
# print(result)

from joblib import dump, load
dump(svm_clf, 'Newmodel.joblib')

clf = load('Newmodel.joblib') 
result = clf.score(test_features, test_labels)
print(result)


# print('Precision: ',metrics.accuracy_score(label2, y_svm_pred))
# print("Recall:",metrics.recall_score(label2, y_svm_pred))

# print('____________')

# # dự đoán cho 1 newinput
# img2 = np.asarray(plt.imread('ap.jpg'))
# img2_cvt = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# img2_pre = resize(img2_cvt, (100, 100),anti_aliasing=True)
# img2_Ex = FtrExtractHOG(img2_pre)
# newtest = []
# newtest.append(img2_Ex)

# a = svm_clf.predict(newtest)
# s = a[0]

# prob = svm_clf.predict_proba(newtest)
# print('Ket Qua Du Doan:', Dict[s] )
# print('Kha Nang:', prob)



