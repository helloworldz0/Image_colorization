# Aditya's Improved Version
import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  # Use the appropriate backend for your environment
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

# --------------------------------------------------
# Predict and visualize result
# --------------------------------------------------
model = load_model('best_model_faces.keras',compile=False)
folder_path = './Data/Grayscale_2/'
cam=cv2.VideoCapture(0)
ret,frame=cam.read()
cam.release()
cv2.imwrite('165.jpg',frame)
img = '165.jpg'

width, height = PILImage.open(img).size
print(width, height)

gray_img = load_img(img, color_mode='grayscale', target_size=(256,256))
gray_img = img_to_array(gray_img).astype('float32') / 255.0
X_test = np.reshape(gray_img, (1, 256, 256, 1))

output = model.predict(X_test)
output = np.reshape(output, (256, 256, 2))
output = cv2.resize(output, (width, height), interpolation=cv2.INTER_CUBIC)

outputLAB = np.zeros((height, width, 3), dtype=np.float32)
img_resized = cv2.resize(np.reshape(gray_img, (256, 256)), (width, height), interpolation=cv2.INTER_CUBIC)

# Denormalize: L*100, a,b*128
outputLAB[:, :, 0] = img_resized * 100.0
outputLAB[:, :, 1:] = output * 128.0

# Clip to valid LAB ranges
outputLAB[:, :, 0] = np.clip(outputLAB[:, :, 0], 0, 100)
outputLAB[:, :, 1:] = np.clip(outputLAB[:, :, 1:], -128, 127)

rgb_image = lab2rgb(outputLAB)

plt.subplot(1,2,1)
plt.imshow(img_resized, cmap='gray')
plt.title('Input (Grayscale)')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(rgb_image)
plt.title('Colorized Output')
plt.axis('off')
plt.show()
