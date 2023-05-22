""" Paquete para adición de ruido gaussiano a imágenes.
Este script le permite al usuario aplicar ruido gaussiano
a imagenes almacenadas en S3.

Este archivo puede importarse como modulo y contiene las siguientes funciones:

    * list_objects: lista y devuelve los objetos que se encuentran en un 
                    bucket de Amazon S3 y coinciden con un prefijo 
                    específico.
    * add_noise: aplica un efecto de desenfoque gaussiano a una imagen.
    * gaussian_noise: agrega un efecto de ruido gaussiano a 
                    imágenes almacenadas en S3.
"""

import yaml
import boto3
from typing import List
from PIL import Image, ImageFilter
import io
import numpy as np
import cv2 as cv

# Abrir yaml para obtener variables globales
with open("configs/config.yaml", encoding="utf-8") as file:
    config = yaml.safe_load(file)
file.close()

# Variables globales
PREPROCESSED_TRAIN_PREFIX = config['gaussian']['PREPROCESSED_TRAIN_PREFIX']
PREPROCESSED_VALID_PREFIX = config['gaussian']['PREPROCESSED_VALID_PREFIX']
GAUSSIAN_TRAIN_PREFIX = config['gaussian']['GAUSSIAN_TRAIN_PREFIX']
GAUSSIAN_VALID_PREFIX = config['gaussian']['GAUSSIAN_VALID_PREFIX']
BUCKET_NAME = config['aws_config']['BUCKET_NAME']
S3_PROFILE = config['aws_config']['PROFILE_NAME']

session = boto3.Session(profile_name=S3_PROFILE)
s3_client = session.client('s3')

def list_objects(client = None, bucket_name: str = None, prefix: str = None):
    """
    Función para listar los objetos que se encuentran en un 
    bucket de Amazon S3 y coinciden con un prefijo específico.
    ----------------------------------------------------------
    Inputs:
        client (boto3.client): Cliente de Amazon S3 para interactuar 
                               con el servicio.
        bucket_name (str): Nombre del bucket de Amazon S3.
        prefix (str): Prefijo utilizado para filtrar los objetos 
                      del bucket.
    
    Outputs: 
        s3_objects (list): Lista de objetos que coinciden con el 
                           prefijo especificado.
    """
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

def put_image_s3(
                 client = None, 
                 bucket_name: str = None,
                 prefix: str = None,
                 img_name: str = None,
                 image: np.array = None
            ):
    """
    Función para guardar una imagen en un bucket de Amazon S3.
    --------------------------------------------------------
    Inputs:
        client (boto3.client): Cliente de Amazon S3 para interactuar 
                                con el servicio.
        bucket_name (str): Nombre del bucket de Amazon S3.
        prefix (str): Prefijo utilizado para organizar las imágenes 
                      en el bucket.
        img_name (str): Nombre de la imagen a guardar en el bucket.
        image (np.array): Arreglo que representa la imagen 
                          a guardar.
    
    Outputs: 
        None
    """
    # codificar la imagen Numpy a png
    _, im_buff_arr = cv.imencode(".png", image)
    # regresar la imagen de Numpy a bytes para guardar en S3
    byte_im = im_buff_arr.tobytes()
    name = f'{prefix}/{img_name}'
    print(name)
    # guardar la imagen en el bucket con el nombre creado
    client.put_object(Bucket = bucket_name, Body = byte_im, Key = name)

def add_noise(
              object: str = None,
              image: Image = None,
              radius: int = 2.3
            ):
    # mantener únicamente el nombre y descartar el prefijo
    img_name = object.split('/')[-1]
    # agregar '_blured' al nombre de la imagen
    new_img_name = img_name[:-4] + "_blured" + img_name[-4:]
    # aplicar el ruido gaussiano a la imagen con ayuda de PIL
    blured_img = image.filter(ImageFilter.GaussianBlur(radius = radius))
    return new_img_name, blured_img


def gaussian_noise(
                    client = None,
                    bucket_name: str = None,
                    prefix: str = None,
                    objects: List[str] = []
                ):
    """
    Función para agregar ruido gaussiano a varias imágenes 
    almacenadas en un bucket de Amazon S3.
    -------------------------------------------------------------------------------------------
    Inputs:
        client (boto3.client): Cliente de Amazon S3 para 
                               interactuar con el servicio.
        bucket_name (str): Nombre del bucket de Amazon S3.
        prefix (str): Prefijo utilizado para organizar las 
                      imágenes en el bucket.
        objects (List[str]): Lista de nombres de objetos 
        (imágenes) en el bucket a las que se les agregará 
         ruido.
    
    Outputs: 
        True (bool): Indica que se completó exitosamente el 
                     proceso de agregar ruido a las imágenes 
                     y guardarlas en S3.
    """
    for object in objects:
        # obtener la imagen de S3 en bytes
        response = client.get_object(Bucket = bucket_name, Key = object)
        image = response['Body'].read()
        # leer la imagen en formato PIL
        image_pil = Image.open(io.BytesIO(image))
        new_img_name, image_blured = add_noise(object, image_pil)
        # convertir la imagen con ruido de PIL a Numpy
        res_image = np.asarray(image_blured)
        # guardar objeto en S3
        put_image_s3(client, bucket_name, prefix, new_img_name, res_image)
    return True

if __name__ == '__main__':
    # listar las imágenes en el bucket y con el prefijo dados
    train_objects = list_objects(s3_client, BUCKET_NAME, PREPROCESSED_TRAIN_PREFIX)
    # mantener únicamente el nombre de la imagen
    train_objects = [obj['Key'] for obj in train_objects]

    # listar las imágenes en el bucket y con el prefijo dados
    valid_objects = list_objects(s3_client, BUCKET_NAME, PREPROCESSED_VALID_PREFIX)
    # mantener únicamente el nombre de la imagen
    valid_objects = [obj['Key'] for obj in valid_objects]

    # agregar ruido gaussiano a las imágenes de entrenamiento y guardarlas en S3
    print(gaussian_noise(s3_client, BUCKET_NAME, GAUSSIAN_TRAIN_PREFIX, train_objects))
    # agregar ruido gaussiano a las imágenes de validación y guardarlas en S3
    print(gaussian_noise(s3_client, BUCKET_NAME, GAUSSIAN_VALID_PREFIX, valid_objects))
