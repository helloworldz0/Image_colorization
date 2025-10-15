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

# --------------------------------------------------
# Predict and visualize result
# --------------------------------------------------
model = load_model('./best_model_faces.h5', compile=False)
# folder_path = './Data/Grayscale_2/'
# img = '1664.jpg'
# img = folder_path + img

# Create a VideoCapture object
cap = cv2.VideoCapture(1)
# Capture a single frame
ret, frame = cap.read()
# Release the video capture device
cap.release()

# Crop the image (you can adjust the coordinates)
x, y, w, h = 160, 120, 320, 240
cropped_frame = frame[y:y+h, x:x+w]

# cv2.imwrite('./temp.jpg', frame)
# img = './temp.jpg'

cv2.imwrite('./temp.jpg', cropped_frame)
img = './temp.jpg'

# width, height = PILImage.open(img).size
# print("Width, Height(Uncropped): ", width, height)

width, height = PILImage.open(img).size
print("Width, Height(Cropped): ", width, height)

gray_img = load_img(img, color_mode='grayscale', target_size=(256,256))
gray_img = img_to_array(gray_img).astype('float32') / 255.0
X_test = np.reshape(gray_img, (1, 256, 256, 1))
