from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
 
def createModel():
    model = Sequential()
    model.add(Conv2D(32, (8, 8), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))
     
    return model

    import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os, sys
from os import listdir
from os.path import isfile, join
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def load_data(img_dir):
    return np.array([cv2.resize(cv2.imread(os.path.join(img_dir, img)),(64,64)) for img in os.listdir(img_dir) if img.endswith(".jpeg")])

mypath = "./data/insta/"

igers = [f for f in listdir(mypath) if not isfile(join(mypath, f))]

data_labels_int=[]
data_labels_class=[]
data_image=[]
i=0
for iger in igers:
    data_labels_class.append(iger)
    for img in os.listdir(os.path.join(mypath, iger)):
        if img.endswith(".jpeg"):
            data_labels_int.append(i)
            data_image.append(np.array(cv2.resize(cv2.imread(os.path.join(mypath,iger,img)),(64,64))))
            #data_image.append((i,np.array(cv2.imread(img))))
    i+=1
print data_labels_class       
print str(len(data_image)) + " images"         
print str(len(data_labels_int)) + " labels"
#print data_image[0][1]
#plt.imshow(cv2.cvtColor(data_image[554], cv2.COLOR_BGR2RGB)) #cv2.COLOR_BGR2RGB
#plt.axis("off")

train_images, test_images, train_labels_int, test_labels_int= train_test_split(np.array(data_image),data_labels_int, test_size=.25)
#train_labels = tf.one_hot(train_labels_int,len(data_labels_class))
#test_labels = tf.one_hot(test_labels_int,len(data_labels_class))
train_labels=np.array(train_labels_int)
test_labels=np.array(test_labels_int)
#print str(len(train_data)) + " images for train"
#print str(train_labels.shape) + " labels"
#print str(len(test_data)) + " images for test"
#print str(test_labels.shape) + " labels"

print('Training data shape : ', train_images.shape, train_labels.shape)

# Find the shape of input images and create the variable input_shape
nRows,nCols,nDims = train_images.shape[1:]
train_data = train_images.reshape(train_images.shape[0], nRows, nCols, nDims)
test_data = test_images.reshape(test_images.shape[0], nRows, nCols, nDims)
input_shape = (nRows, nCols, nDims)

# Change to float datatype
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

# Scale the data to lie between 0 to 1
train_data /= 255
test_data /= 255

# Change the labels from integer to categorical data
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)


#print data_labels_int[544]




#model = createModel()
#model.summary()

nClasses=len(igers)


model1 = createModel()
batch_size = 20
epochs = 10
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model1.summary()

history = model1.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1, 
                   validation_data=(test_data, test_labels_one_hot))
model1.evaluate(test_data, test_labels_one_hot)