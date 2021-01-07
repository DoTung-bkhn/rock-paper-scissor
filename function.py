import os,sys
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
#np.set_printoptions(threshold=sys.maxsize)   #print full numpy array

#loading image
def load_image(path):
    dataset=[];image_path=[]
    for i in os.listdir(path):
            img_path=os.path.join(path,i)
            image_path.append(img_path)
    for i in range(len(image_path)):
        image=cv.imread(image_path[i])
        img_resize=cv.resize(image,(150,150))
        dataset.append(img_resize)
    return dataset

#labelize data
def label(rock_num,paper_num,scisssor_num):
    rock=np.zeros((rock_num,1),dtype=int)
    paper=np.ones((paper_num,1),dtype=int)
    scissor=np.full((scisssor_num,1),2,dtype=int)
    target=np.vstack((rock,paper,scissor))
    return target

#create model
def create_model():
    model=tf.keras.models.Sequential()
    model.add(Conv2D(64,(3,3),activation='relu',input_shape=(150,150,3)))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(64, (3, 3),activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3),activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dropout(0.9))
    model.add(Dense(units=512,activation='relu'))
    model.add(Dense(units=3,activation='softmax'))
    model.summary()
    model.compile(optimizer='Adam',loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])
    return model

#save model
def save_model(model):
    model.save('C:/Users/ADMIN/Desktop/python/rock paper scissors/my_model.h5',overwrite=True)

#predict
def predict(model,input):
    pos=[]
    output=model.predict(input,batch_size=len(input))
    for i in range(len(output)):
        max_val=max(output[i])
        for i,j in enumerate(output[i]):
            if j==max_val:position=i
        pos.append(position)
    return pos


