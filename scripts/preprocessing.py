""" Paquete para preprocesamiento de datos.
Este script le permite al usuario hacer un procesamiento de los datos
de entrenamiento y prueba.
Al modificar este codigo, otro preprocesamiento puede ser especificado.

Este archivo puede importarse como modulo y contiene las siguientes funciones:

    * list_objects: lista y devuelve los objetos que se encuentran en un 
                    bucket de Amazon S3 y coinciden con un prefijo 
                    específico.
    * prop_black_pixels: calcula la proporción de píxeles negros 
                         en un imagen.
    * put_image_s3: Redimensiona imagen y subir a bucket de S3.
    * preprocess_image: preprocesa imagen y subida a un bucket de S3.
    * preprocess_images: preprocesamiento de varias imágenes 
                         almacenadas en un bucket de S3.
"""

import yaml
import boto3
from typing import List
from PIL import Image
import io
import numpy as np
import cv2 as cv

# Abrir yaml
with open("configs/config.yaml", encoding="utf-8") as file:
    config = yaml.safe_load(file)
file.close()

# Variables globales
RAW_TRAIN_PREFIX = config['preprocess']['RAW_TRAIN_PREFIX']
RAW_VALID_PREFIX = config['preprocess']['RAW_VALID_PREFIX']
PREPROCESSED_TRAIN_PREFIX = config['preprocess']['PREPROCESSED_TRAIN_PREFIX']
PREPROCESSED_VALID_PREFIX = config['preprocess']['PREPROCESSED_VALID_PREFIX']
BUCKET_NAME = config['aws_config']['BUCKET_NAME']
S3_PROFILE = config['aws_config']['PROFILE_NAME']

session = boto3.Session(profile_name=S3_PROFILE)
s3_client = session.client('s3')

def list_objects(client = None, bucket_name: str = None, prefix: str = None):
    # crear lista donde se almacenan las _uri_ de cada imagen
    s3_objects = []
    if client:
        try:
            # listar las imágenes que se encuentran en el bucket dado y que coinciden
            # con el prefijo
            s3_response = client.list_objects_v2(Bucket = bucket_name, Prefix = prefix)
            s3_objects.extend(s3_response['Contents'])
            # en caso de que haya varias páginas de imágenes, iterar sobre cada una
            while s3_response['IsTruncated']:
                # solicitar el siguiente iterador correspondiente a la siguiente página
                next_token = s3_response['NextContinuationToken']
                # listar los objetos que se encuentren en la página actual
                s3_response = client.list_objects_v2(
                                                     Bucket = bucket_name,
                                                     Prefix = prefix,
                                                     ContinuationToken = next_token
                                                )
                if 'Contents' in s3_response:
                    s3_objects.extend(s3_response['Contents'])
            return s3_objects
        except Exception:
            raise Exception

def prop_black_pixels(image):
    # evalúa la proporción de pixeles con valores menores o igual a 30
    num_black_pixels = np.sum(image <= 30)
    return num_black_pixels / image.size

def put_image_s3(
                 client = None, 
                 bucket_name: str = None,
                 prefix: str = None,
                 image: np.array = None,
                 count_img: int = 0
            ):
    # reescalar a un tamaño predefinido
    tmp_resized = cv.resize(image, (400,512), interpolation = cv.INTER_AREA)
    # codificar la imagen Numpy a png
    _, im_buff_arr = cv.imencode(".png", tmp_resized)
    # regresar la imagen de Numpy a bytes para guardar en S3
    byte_im = im_buff_arr.tobytes()
    # identificar si es imagen de entrenamiento o validación
    folder = prefix.split('/')[-1]
    name = f'{prefix}/{count_img}_{folder}.png'
    # guardar la imagen en el bucket con el nombre creado
    client.put_object(Bucket = bucket_name, Body = byte_im, Key = name)
    print(name)

def preprocess_image(
                    client = None, 
                    bucket_name: str = None,
                    prefix: str = None,
                    image: np.array = None,
                    count_img: int = 0
                    ):
    response = 0
    tmp_image = image.copy()
    # obtener la proporción de pixeles negros que posee la imagen
    pbp = prop_black_pixels(tmp_image)
    # descartar la imagen si la proporción de pixeles negros es mayor
    # que la proporción de pixeles de la radiografía
    if pbp <= 0.6:
        img_shape = tmp_image.shape
        h = img_shape[0]
        w = img_shape[1]
        # si la imagen está rotada
        if w > h:
            tmp_image = cv.rotate(tmp_image, cv.ROTATE_90_CLOCKWISE)
        # descargar la imagen si está muy delgada
        if w > 250:
            # guardar la imagen en S3
            put_image_s3(client, bucket_name, prefix, tmp_image, count_img)
            response = 1
    return response

def preprocess_images(
                      client = None,
                      bucket_name: str = None,
                      prefix: str = None,
                      objects: List[str] = []
                    ):
    count_img = 0
    for object in objects:
        # obtener la imagen de S3 en bytes
        response = client.get_object(Bucket = bucket_name, Key = object)
        image = response['Body'].read()
        # leer la imagen en formato PIL
        image = Image.open(io.BytesIO(image))
        # convertir la imagen con ruido de PIL a Numpy
        image = np.asarray(image)
        # preprocesar la imagen, evaluar si es imagen candidata, y guardarla en caso positivo
        was_image_saved = preprocess_image(client, bucket_name, prefix, image, count_img)
        count_img += was_image_saved
    return count_img

if __name__ == '__main__':
    # listar las imágenes en el bucket y con el prefijo dados
    train_objects = list_objects(s3_client, BUCKET_NAME, RAW_TRAIN_PREFIX)
    # mantener únicamente el nombre de la imagen
    train_objects = [obj['Key'] for obj in train_objects]

    # listar las imágenes en el bucket y con el prefijo dados
    valid_objects = list_objects(s3_client, BUCKET_NAME, RAW_VALID_PREFIX)
    # mantener únicamente el nombre de la imagen
    valid_objects = [obj['Key'] for obj in valid_objects]

    print(preprocess_images(s3_client, BUCKET_NAME, PREPROCESSED_TRAIN_PREFIX, train_objects))
    print(preprocess_images(s3_client, BUCKET_NAME, PREPROCESSED_VALID_PREFIX, valid_objects))

    # response = s3_client.get_object(Bucket = BUCKET_NAME, Key = 'preprocessed/valid/0_valid.png')
    # image_content = response['Body'].read()
    # image = Image.open(io.BytesIO(image_content))
    # image = np.asarray(image)
    # print(image)
