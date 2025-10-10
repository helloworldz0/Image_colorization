# Aditya's Version
from IPython.display import display, Image
from matplotlib.pyplot import imshow
from keras.layers import Conv2D, UpSampling2D, InputLayer, Reshape
from keras.models import Sequential, load_model
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import lab2rgb, rgb2lab
from skimage import color
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import keras as keras
import tensorflow as tf
import glob
import cv2 as cv2
import os
import pdb
import PIL

folder_path='./Data/Black_White/' 
images1 = []
for img in os.listdir(folder_path):
    img=folder_path+img
    img = load_img(img, target_size=(256,256)) 
    img = img_to_array(img)/ 255
    X= color.rgb2gray(img)
    images1.append(X)
#pdb.set_trace()

folder_path='./Data/colored/' 
images2 = []
for img in os.listdir(folder_path):
    #print(folder_path+img)
    img=folder_path+img
    img = load_img(img, target_size=(256,256)) 
    img = img_to_array(img)/ 255
    lab_image = rgb2lab(img)
    lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]
    # The input will be the black and white layer
    Y = lab_image_norm[:,:,1:]

    images2.append(Y)
#pdb.set_trace()

X = np.array(images1)
Y = np.array(images2)
#pdb.set_trace()

x1 = keras.Input(shape=(None, None, 1))

x2 = Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)(x1)
x3 = Conv2D(16, (3, 3), activation='relu', padding='same')(x2)
x4 = Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(x3)
x5 = Conv2D(32, (3, 3), activation='relu', padding='same')(x4)
x6 = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(x5)
x7 = UpSampling2D((2, 2))(x6)
x8 = Conv2D(32, (3, 3), activation='relu', padding='same')(x7)
x9 = UpSampling2D((2, 2))(x8)
x10 = Conv2D(16, (3, 3), activation='relu', padding='same')(x9)
x11 = UpSampling2D((2, 2))(x10)
x12 = Conv2D(2, (3,3), activation='sigmoid', padding='same')(x11)

# x12=tf.reshape(x12,(104,104,2))
# x12 = tf.image.resize(x12,[100, 100])
# x12=tf.reshape(x12,(1,100, 100,2))
x12 = tf.keras.layers.Resizing(256, 256, interpolation='bilinear')(x12)

# Finish model
model = keras.Model(x1, x12)

model.compile(optimizer='rmsprop', loss='mse')
model.fit(X,Y, batch_size=8, epochs=2000, verbose=1)

model.evaluate(X, Y, batch_size=1)
model.save('model.h5', True, False)