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