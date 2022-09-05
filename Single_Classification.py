#%% import libraries :
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import livelossplot as llp 
import seaborn as sns 
from tensorflow import keras 
from keras.models import Sequential , Model 
from keras.utils import to_categorical , plot_model 
from keras.layers import Dense , Flatten , Dropout , BatchNormalization , Conv2D , MaxPool2D , Input
from keras.layers import RandomFlip , RandomContrast , RandomZoom ,RandomRotation
from keras.datasets import cifar10 
from sklearn.metrics import confusion_matrix 
import pandas as pd

# %% Load the data :
(x_train , y_train) , (x_test , y_test) = cifar10.load_data()
num_classes=10

#%% Pre-processing the data : 
x_train=x_train.astype("float32") / 255.0
x_test=x_test.astype("float32") / 255.0

#%% One-hot-encoding : 
y_train = to_categorical(y_train , num_classes=num_classes)
y_test = to_categorical(y_test , num_classes=num_classes)

#%% build a model : 
input = Input(shape=(32,32,3))
x= Conv2D(64 , (3,3) , padding="valid" , activation="relu")(input) 
x=Dropout(0.2)(x)
x=BatchNormalization()(x)
x= Conv2D(128 , (3,3) , padding="valid" , activation="relu")(x) 
x= MaxPool2D()(x)
x=Dropout(0.2)(x)
x=BatchNormalization()(x)
x= Conv2D(128 , (3,3) , padding="valid" , activation="relu")(x) 
x= MaxPool2D()(x)
x=Dropout(0.2)(x)
x=BatchNormalization()(x)
x= Conv2D(128 , (3,3) , padding="valid" , activation="relu")(x) 
x= MaxPool2D()(x)
x=Dropout(0.2)(x)
x=BatchNormalization()(x)
x= Flatten()(x)
x= Dense(32 , activation="relu")(x)
output = Dense(10 , activation="softmax")(x)

model=Model(inputs = input , outputs = output) 
model.summary() 
plot_model(model , "fcn_cifar10.png" , show_shapes=True)

# %% Compile and fit the model :
plot_loss = llp.PlotLossesKeras()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history=model.fit(x_train, y_train, batch_size=128,
 epochs=15, validation_data=(x_test,y_test), callbacks=[plot_loss])
