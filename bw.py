import numpy as np # linear algebra
import matplotlib.pyplot as plt
from IPython.display import display, Image
from matplotlib.pyplot import imshow
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import lab2rgb, rgb2lab
from skimage import color
import keras as keras
import tensorflow as tf
import glob
import cv2 as cv2
import os
import pdb
folder_path='./coloured/' 
images2 = []
for img in os.listdir(folder_path):
    #print(folder_path+img)
    img=folder_path+img
    img = load_img(img, target_size=(1000,1000)) 
    img = img_to_array(img)/ 255
    Y = color.rgb2gray(img)
    # The input will be the black and white layer
    images2.append(Y)

cv2.imshow("My Image Window", Y)
cv2.waitKey(0)