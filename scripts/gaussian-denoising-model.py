import argparse
import json
import os
import boto3
import io
from typing import List
from PIL import Image

import numpy as np
import tensorflow as tf


# Definir la arquitectura del autoencoder
def build_autoencoder():
    # Encoder
    input_img = Input(shape=(64, 48, 3))
    h = BatchNormalization()(input_img)

    h = Conv2D(32, (3, 3), padding='same', activation='relu')(h)
    h = MaxPool2D((2, 2))(h)

    h = Conv2D(64, (3, 3), padding='same', activation='elu')(h)
    h = MaxPool2D((2, 2))(h)

    h = Conv2D(128, (3, 3), padding='same', activation='elu')(h)

    encoded = Conv2D(32, (3, 3), padding='same', activation='elu')(h)
    h = Conv2D(32, (3, 3), padding='same', activation='elu')(encoded)

    h = UpSampling2D((2, 2))(h)
    h = Conv2D(128, (3, 3), padding='same', activation='elu')(h)
    h = Conv2D(64, (3, 3), padding='same', activation='elu')(h)

    h = UpSampling2D((2, 2))(h)
    h = Conv2D(32, (3, 3), padding='same', activation='elu')(h)
    
    h = BatchNormalization()(h)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(h)

    # Autoencoder model
    autoencoder = Model(input_img, decoded)
    return autoencoder

def list_objects(client = None, bucket_name: str = None, prefix: str = None):
        s3_objects = []
        if client:
            try:
                s3_response = client.list_objects_v2(Bucket = bucket_name, Prefix = prefix)
                s3_objects.extend(s3_response['Contents'])
                while s3_response['IsTruncated']:
                    next_token = s3_response['NextContinuationToken']
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


def _load_training_data():
    """Load training data"""
    session = boto3.Session(profile_name='datascientist')
    s3_client = session.client('s3')
    BUCKET_NAME='images-itam-denoising'
    RAW_TRAIN_PREFIX : 'raw/train'
    s3_objects = list_objects(client = s3_client, bucket_name=BUCKET_NAME, prefix= RAW_TRAIN_PREFIX)
    noisy_images = []
    for object in s3_objects:
        response = s3_client.get_object(Bucket = BUCKET_NAME, Key = object)
        image = response['Body'].read()
        image = Image.open(io.BytesIO(image))
        image = np.asarray(image)
        # appendear imagenes
        noisy_images.append(image)   

    return noisy_images

    

def _load_testing_data():
    session = boto3.Session(profile_name='datascientist')
    s3_client = session.client('s3')
    BUCKET_NAME='images-itam-denoising'
    RAW_VALID_PREFIX : 'raw/valid'
    s3_objects = list_objects(client = s3_client, bucket_name=BUCKET_NAME, prefix= RAW_VALID_PREFIX)
    noisy_images_valid = []
    for object in s3_objects:
        response = s3_client.get_object(Bucket = BUCKET_NAME, Key = object)
        image = response['Body'].read()
        image = Image.open(io.BytesIO(image))
        image = np.asarray(image)
        # appendear imagenes
        noisy_images_valid.append(image)        

    return noisy_images_valid


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS")))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST"))

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()

    train_data = _load_training_data(args.train)
    eval_data = _load_testing_data(args.train)


    ## ESTO ES LO QUE HAY QUE CAMBIAR:
    mnist_classifier = model(train_data, train_labels, eval_data, eval_labels)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001' in Tensorflow SavedModel Format
        # To export the model as h5 format use model.save('my_model.h5')
        mnist_classifier.save(os.path.join(args.sm_model_dir, "000000001"))