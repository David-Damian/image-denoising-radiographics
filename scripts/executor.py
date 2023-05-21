import os
import numpy as np
import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow
import boto3 

# session = sagemaker.Session(boto3.session.Session())")
sagemaker_session = sagemaker.Session(boto3.session.Session())
role = "arn:aws:iam::345921935563:role/service-role/SageMaker-mldev"

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

autoencoder.fit()



