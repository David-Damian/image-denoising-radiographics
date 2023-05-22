from tensorflow import keras
import numpy as np
import cv2 as cv
import io
from PIL import Image

def get_model():
    return keras.models.load_model('autoencoder.h5')

def preprocessed_images(image):
    OW, NW = 48, 48
    OH, NH = 64, 64
    image = Image.open(io.BytesIO(image))
    image = np.asarray(image)
    print(image)
    images_valid = []
    image = cv.resize(image, (NW, NH), interpolation = cv.INTER_AREA)
    # Normalizar los valores de píxeles en el rango [0, 1]
    image = image.astype(np.float32) / 255.0

    # Agregar las imágenes a las listas
    images_valid.append(image)
    images_valid = np.expand_dims(images_valid, axis=-1)
    images_valid = np.squeeze(images_valid)
    return images_valid

def get_prediction(image, autoencoder):
    print("image types")
    print(type(image))

    images_valid = preprocessed_images(image)
    print(type(images_valid), images_valid.shape, images_valid[0].shape)
    print(images_valid)
    return autoencoder.predict(images_valid)
