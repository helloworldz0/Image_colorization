import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from PIL import Image as PILImage
from tensorflow import keras
from tensorflow.keras.layers import Concatenate,Add,Conv2D, UpSampling2D, BatchNormalization, Activation, Input,Dropout # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator # type: ignore
from tensorflow.keras.models import Model, load_model # type: ignore
from sklearn.model_selection import train_test_split
from skimage.color import lab2rgb, rgb2lab
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# --------------------------------------------------
# Load and prepare the grayscale and color images
# --------------------------------------------------
gray_folder = './Data/Grayscale_2/'
color_folder = './Data/Colored_2/'

gray_files = sorted([f for f in os.listdir(gray_folder) if os.path.isfile(os.path.join(gray_folder, f))])
color_files = sorted([f for f in os.listdir(color_folder) if os.path.isfile(os.path.join(color_folder, f))])

common_files = [f for f in gray_files if f in color_files]
common_files = sorted(common_files)

images1 = []
images2 = []
for fname in common_files:
    gray_path = os.path.join(gray_folder, fname)
    color_path = os.path.join(color_folder, fname)

    # load color image and compute LAB
    color_img = load_img(color_path, target_size=(256,256))
    color_img = img_to_array(color_img).astype('float32') / 255.0
    lab_image = rgb2lab(color_img)

    # L in [0,1], ab in [-1,1]
    L = lab_image[:, :, 0] / 100.0
    ab = lab_image[:, :, 1:] / 128.0

    # Use the L from the color image (keeps pairing exact)
    images1.append(np.expand_dims(L, axis=-1))
    images2.append(ab)

# Convert to numpy arrays
X = np.array(images1, dtype=np.float32)
Y = np.array(images2, dtype=np.float32)

# Ensure correct shape for CNN input
if X.ndim == 3:
    X = np.expand_dims(X, axis=-1)

print("X dtype, min/max:", X.dtype, X.min(), X.max())
print("Y dtype, min/max:", Y.dtype, Y.min(), Y.max())
print("X shape, Y shape:", X.shape, Y.shape)
def residual_block(x, filters, kernel_size=3, dropout_rate=0.2):
    """Two Conv2D layers with BatchNorm, ReLU, optional Dropout, and a skip (residual) connection."""
    shortcut = x
    x = Conv2D(filters, (kernel_size, kernel_size), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2D(filters, (kernel_size, kernel_size), padding='same')(x)
    x = BatchNormalization()(x)

    # If the number of filters differs, match dimensions for addition
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


# === Encoder ===
inputs = Input(shape=(256, 256, 1))

# Encoder Block 1
x1 = Conv2D(16, (3, 3), strides=2, padding='same')(inputs)  # 128x128x16
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x1 = residual_block(x1, 16)

# Encoder Block 2
x2 = Conv2D(32, (3, 3), strides=2, padding='same')(x1)      # 64x64x32
x2 = BatchNormalization()(x2)
x2 = Activation('relu')(x2)
x2 = residual_block(x2, 32)

# Encoder Block 3
x3 = Conv2D(64, (3, 3), strides=2, padding='same')(x2)      # 32x32x64
x3 = BatchNormalization()(x3)
x3 = Activation('relu')(x3)
x3 = residual_block(x3, 64)

# Bottleneck
x4 = residual_block(x3, 128, dropout_rate=0.3)               # 32x32x128


# === Decoder ===
# Up 1
x5 = UpSampling2D((2, 2))(x4)                                # 64x64x128
x5 = Concatenate()([x5, x2])
x5 = residual_block(x5, 64, dropout_rate=0.2)

# Up 2
x6 = UpSampling2D((2, 2))(x5)                                # 128x128x64
x6 = Concatenate()([x6, x1])
x6 = residual_block(x6, 32, dropout_rate=0.2)

# Up 3 (to original resolution)
x7 = UpSampling2D((2, 2))(x6)                                # 256x256x32
x7 = residual_block(x7, 16, dropout_rate=0.1)

# Output layer (2 channels, normalized)
outputs = Conv2D(2, (3, 3), activation='sigmoid', padding='same')(x7)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse')

model.summary()
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=10, zoom_range=0.1)

callbacks = [
    keras.callbacks.ModelCheckpoint('testscheckpoint.keras', save_best_only=True, monitor='val_loss'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

hist=model.fit(
    datagen.flow(X_train, Y_train, batch_size=1, shuffle=True, seed=seed),
    epochs=5,
    verbose=2,
    validation_data=(X_val, Y_val),
    callbacks=callbacks
)
model.evaluate(X_val, Y_val, batch_size=1, verbose=2)
model.save('tests.keras')
