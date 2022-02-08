import glob
import cv2
import numpy as np
from PIL import Image
import splitfolders
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# read the image file of Normal (negative)
file = r'C:\Users\EGYPT\PycharmProjects\pythonProject1\Dataset\Negative\*.jpg'
path_copy_Negative = 'C:\\Users\\EGYPT\PycharmProjects\\pythonProject1\\data\\NegatibeBirized\\'
i =0
thresh = 200
maxval = 255
#birization of negative images and save it
for fn in glob.glob(file):
    iml = Image.open(fn)
    im_gray = np.array(Image.open(fn).convert('L'))
    i+=1
    im_bin = (im_gray > thresh) * maxval
    Image.fromarray(np.uint8(im_bin)).save(path_copy_Negative+f'bin{i}.jpg')

i=0
# read the image file of (positive)
file = r'C:\Users\EGYPT\PycharmProjects\pythonProject1\Dataset\Positive\*.jpg'
path_copy_Positive ='C:\\Users\EGYPT\\PycharmProjects\\pythonProject1\\data\\PositiveBinrized\\'
#birization of negative images and save it
for fn in glob.glob(file):
    iml = Image.open(fn)
    im_gray = np.array(Image.open(fn).convert('L'))
    i+=1
    im_bin = (im_gray > thresh) * maxval
    Image.fromarray(np.uint8(im_bin)).save(path_copy_Positive+f'bin{i}.jpg')


input_folder = 'data/'
#splitting data into 80% test and 20% training on folders
splitfolders.ratio(input_folder,output="images",
                   seed=3,ratio=(0.8,0.0,0.2),group_prefix=None)
#prepare data
path = os.listdir('C:\\Users\\EGYPT\\PycharmProjects\\pythonProject1\\images\\train')
classes = {'NegatibeBirized': 0, 'PositiveBinrized': 1}

X = []
Y = []
for clc in classes:
    pth = 'C:\\Users\\EGYPT\\PycharmProjects\\pythonProject1\\images\\train\\'+clc
    for k in os.listdir(pth):
        img = cv2.imread(pth+'\\'+k, 0)
        X.append(img)
        Y.append(classes[clc])


X = np.array(X)
Y = np.array(Y)

X_update = X.reshape(len(X),-1) #to reshape X for SVM ,X.shape = (nrow,100*100)

# split data to test and training
xtrain , xtest ,ytrain ,ytest = train_test_split(X_update,Y,random_state=0,test_size=0.2)

# scaling features to make it within 0 to 1
xtrain = xtrain / maxval
xtest = xtest / maxval

sv = SVC()
sv.fit(xtrain,ytrain)
print ("Test scores of rbf kernel : ", sv.score(xtest,ytest)*100)

#Prediction
pred = sv.predict(xtest)
print (classification_report(ytest,pred))
sv = SVC(kernel='linear')
sv.fit(xtrain,ytrain)
print ("Test scores of linear kernel : ", sv.score(xtest,ytest)*100)
#Prediction
pred = sv.predict(xtest)
print (classification_report(ytest,pred))

sv = SVC(kernel='poly')
sv.fit(xtrain,ytrain)
print ("Test scores of linear : ", sv.score(xtest,ytest)*100)
#Prediction
pred = sv.predict(xtest)
print (classification_report(ytest,pred))

sv = SVC(kernel='sigmoid')
sv.fit(xtrain,ytrain)
print ("Test scores of sigmoid kernel : ", sv.score(xtest,ytest)*100)
#Prediction
pred = sv.predict(xtest)
print (classification_report(ytest,pred))


param ={'kernel':('linear','sigmoid','poly','rbf'),
        'C':[0.1,1,10,100],
        'gamma':[1,0.1,0.001,0.0001]
        }

sv = SVC()
grids = GridSearchCV(sv,param)
grids.fit(xtrain,ytrain)
print (grids.best_params_)














