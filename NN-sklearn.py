# import the necessary packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
from PIL import Image
import splitfolders
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neural_network import MLPClassifier
import cv2

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

xtrain = xtrain/maxval
xtest = xtest/maxval

nn = MLPClassifier(activation = "logistic",solver="sgd",hidden_layer_sizes=(10,15),random_state=1)
nn.fit(xtrain,ytrain)
pred = nn.predict(xtest)
scores = nn.score(xtest,ytest)
print("Accuracy for logistic activation :",scores)


pp = MLPClassifier(activation = "relu",solver="adam",hidden_layer_sizes=(10,15),random_state=0)
pp.fit(xtrain,ytrain)
pred = nn.predict(xtest)
scores = nn.score(xtest,ytest)
print("Accuracy for relu activation: ",scores)


