"""
Modulo para entrenar un modelo de denoising de imágenes utilizando SageMaker.
"""

import os
import numpy as np
import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow
import boto3 

# Configura la sesión de SageMaker
sagemaker_session = sagemaker.Session(boto3.session.Session())
role = "arn:aws:iam::345921935563:role/service-role/SageMaker-mldev"

# Creación de la instancia de TensorFlow en SageMaker
autoencoder = TensorFlow(
    entry_point="gaussian-denoising-model.py",
    role=role,
    instance_count=2,
    instance_type="ml.m5.xlarge",
    framework_version="2.1.0",
    py_version="py3",
    distribution={"parameter_server": {"enabled": True}},
    output_path="s3://images-itam-denoising/model",
)
# inicia entrenamiento del modelo
autoencoder.fit()



