"""
Script para entrenar un modelo de autoencoder en Amazon SageMaker. 
Este modelo se utiliza para procesar imágenes, reconstruyendo imágenes con ruido 
a partir de imágenes sin ruido. El script carga los datos de entrenamiento desde 
un bucket de Amazon S3 y utiliza TensorFlow para construir y entrenar el modelo 
de autoencoder. 

El modelo resultante puede utilizarse para denoising de imágenes.
"""

import argparse
import json
import os
import boto3
import io
import cv2 as cv
import numpy as np
import tensorflow as tf
import logging

from typing import List
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Dropout, LeakyReLU

logging.basicConfig(level=logging.INFO)


OW, NW = 48, 48
OH, NH = 64, 64

TRAIN_SIZE = 1000

# Definir la arquitectura del autoencoder
def build_autoencoder(noisy_images, original_images):
    """
    Construye un modelo de autoencoder para el procesamiento de imágenes.
    El modelo se compone de capas de codificación y decodificación para 
    reconstruir imágenes con ruido añadido
    a partir de imágenes sin ruido.

    Inputs:
        - noisy_images: Imágenes con ruido que se reconstruiran con este modelo
        - original_images: Imágenes originales sin ruido.

    Output:
        - autoencoder: El modelo de autoencoder construido.

    """
    # Encoder
    input_img = Input(shape=(64, 48, 3))
    h = BatchNormalization()(input_img)

    h = Conv2D(32, (3, 3), padding="same", activation="relu")(h)
    h = MaxPool2D((2, 2))(h)

    h = Conv2D(64, (3, 3), padding="same", activation="elu")(h)
    h = MaxPool2D((2, 2))(h)

    h = Conv2D(128, (3, 3), padding="same", activation="elu")(h)

    encoded = Conv2D(32, (3, 3), padding="same", activation="elu")(h)
    h = Conv2D(32, (3, 3), padding="same", activation="elu")(encoded)

    h = UpSampling2D((2, 2))(h)
    h = Conv2D(128, (3, 3), padding="same", activation="elu")(h)
    h = Conv2D(64, (3, 3), padding="same", activation="elu")(h)

    h = UpSampling2D((2, 2))(h)
    h = Conv2D(32, (3, 3), padding="same", activation="elu")(h)

    h = BatchNormalization()(h)
    decoded = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(h)

    # Autoencoder model
    autoencoder = Model(input_img, decoded)
    logging.info(autoencoder.summary())
    logging.info("Compiling model")
    autoencoder.compile(
        optimizer="adam", loss="mean_squared_error", metrics=["accuracy"]
    )
    logging.info("Training model")
    autoencoder.fit(
        noisy_images,
        original_images,
        batch_size=16,
        epochs=250,
        verbose=1,
        validation_split=0.1,
    )
    logging.info("training finished")
    return autoencoder


def list_objects(client=None, bucket_name: str = None, prefix: str = None):
    s3_objects = []
    if client:
        try:
            s3_response = client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            s3_objects.extend(s3_response["Contents"])
            while s3_response["IsTruncated"]:
                next_token = s3_response["NextContinuationToken"]
                s3_response = client.list_objects_v2(
                    Bucket=bucket_name, Prefix=prefix, ContinuationToken=next_token
                )
                if "Contents" in s3_response:
                    s3_objects.extend(s3_response["Contents"])
            return s3_objects
        except Exception:
            raise Exception


def _load_training_data_y():
    """
    Carga los datos de entrenamiento (imágenes originales) 
    desde un bucket de S3.

    Returns:
        - original_images(np.array): Un array que contiene las 
                                     imágenes originales.
    """
    logging.info("Loading training data")
    session = boto3.Session()
    s3_client = session.client("s3")
    BUCKET_NAME = "images-itam-denoising"
    RAW_TRAIN_PREFIX = "preprocessed/train"
    s3_objects = list_objects(
        client=s3_client, bucket_name=BUCKET_NAME, prefix=RAW_TRAIN_PREFIX
    )
    original_images = []
    for object in s3_objects[:TRAIN_SIZE]:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=object["Key"])
        image = response["Body"].read()
        image = Image.open(io.BytesIO(image))
        image = np.asarray(image)
        image = format_image(image, OW, OH)
        original_images.append(image)
    original_images = np.expand_dims(original_images, axis=-1)
    original_images = np.squeeze(original_images)
    logging.info("Loading training finished")
    return original_images


def _load_training_data_x():
    """
    Carga los datos de entrenamiento (imágenes con ruido) 
    desde un bucket de S3.

    Returns:
        - noisy_images(np.array): Arreglo que contiene las imágenes con ruido.
    """
    logging.info("Loading training data noisy")
    session = boto3.Session()
    s3_client = session.client("s3")
    BUCKET_NAME = "images-itam-denoising"
    GAUSSIAN_TRAIN_PREFIX = "gaussian/train"
    s3_objects = list_objects(
        client=s3_client, bucket_name=BUCKET_NAME, prefix=GAUSSIAN_TRAIN_PREFIX
    )
    noisy_images = []
    for object in s3_objects[:TRAIN_SIZE]:

        # logging.debug(f"Loading object {counter}: {object}")
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=object["Key"])
        image = response["Body"].read()
        image = Image.open(io.BytesIO(image))
        image = np.asarray(image)
        image = format_image(image, NW, NH)
        noisy_images.append(image)
    noisy_images = np.expand_dims(noisy_images, axis=-1)
    noisy_images = np.squeeze(noisy_images)

    logging.info("Loading training data noisy finished")
    return noisy_images


def format_image(image, fx, fy):
    """
    Formatea una imagen ajustándola a un tamaño específico y normalizando 
    los valores de píxeles en el rango [0, 1].

    Inputs:
        - image: La imagen a formatear.
        - fx (int): El ancho deseado de la imagen.
        - fy (int): La altura deseada de la imagen.

    Returns:
        - image: La imagen formateada.
    """
    image = cv.resize(image, (fx, fy), interpolation=cv.INTER_AREA)
    # Normalizar los valores de píxeles en el rango [0, 1]
    image = image.astype(np.float32) / 255.0
    if len(image.shape) != 3:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    return image

# ejecuta el código principal del script.
if __name__ == "__main__":
    # args, unknown = _parse_args()
    logging.info("start")
    noisy_images = _load_training_data_x()
    original_images = _load_training_data_y()
    autoencoder = build_autoencoder(
        noisy_images=noisy_images, original_images=original_images
    )
