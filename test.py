# Aditya's Improved Version
import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image as PILImage
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization, Activation, Input # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator # type: ignore
from tensorflow.keras.models import Model, load_model # type: ignore
from sklearn.model_selection import train_test_split
from skimage.color import lab2rgb, rgb2lab
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
model = load_model('best_model_faces.keras',compile=False)
folder_path = './Data/Grayscale_2/'
cam=cv2.VideoCapture(0)
ret,frame=cam.read()
cam.release()
frame=np.array(frame)
print(frame.shape)

width, height =640,480
# frame = keras.utils.array_to_img(frame)
# gray_img = load_img(frame, color_mode='grayscale', target_size=(256,256))
gray_img = frame[192:448,112:368]
X_test = np.reshape(gray_img, (1, 256, 256, 1))
