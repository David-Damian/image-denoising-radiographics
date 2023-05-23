import tensorflow as tf
import numpy as np
import cv2 as cv
import io
from PIL import Image

def get_model():
    return tf.keras.models.load_model('autoencoder.h5')

def preprocessed_images(image):
    OW, NW = 48, 48
    OH, NH = 64, 64
    image = Image.open(io.BytesIO(image))
    image = np.asarray(image)
    images_valid = []
    image = cv.resize(image, (NW, NH), interpolation = cv.INTER_AREA)
    # Normalizar los valores de píxeles en el rango [0, 1]
    image = image.astype(np.float32) / 255.0
    if len(image.shape) != 3:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    # Agregar las imágenes a las listas
    images_valid.append(image)
    images_valid = np.expand_dims(images_valid, axis=-1)
    images_valid = np.squeeze(images_valid)
    return images_valid

def get_prediction(image, autoencoder, single=True):
    images_valid = preprocessed_images(image)
    if single:
        images_valid = tf.image.resize(images_valid, (64, 48))
        images_valid = tf.expand_dims(images_valid, axis=0)
    image = autoencoder.predict(images_valid)
    image = np.squeeze(image[0])

    data = Image.fromarray(image)
    data.save('/code/predict.png', 'RGB')
    # image = image.tobytes()
    return True
