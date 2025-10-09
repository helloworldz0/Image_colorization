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

model = load_model('./model.h5')
folder_path='./Data/Test/' 
img='horse.jpg'
img=folder_path+img

width, height = PIL.Image.open(img).size
print(width, height)
img = load_img(img, color_mode = "grayscale") 
img = img_to_array(img)/ 255

X = np.array(img)
X = np.expand_dims(X, axis=2)
X = np.reshape(X,(1,height,width,1))
output = model.predict(X)
output=np.reshape(output,(100,100,2))
output=cv2.resize(output,(width,height))
AB_img = output
outputLAB = np.zeros((height,width, 3))
img=np.reshape(img,(height,width))
outputLAB[:,:,0] = img
outputLAB[:,:,1:] = AB_img
outputLAB = (outputLAB * [100, 255, 255]) - [0, 128, 128]
rgb_image = lab2rgb(outputLAB)

imshow(rgb_image)
plt.show()