import os
from matplotlib import pyplot as plt
import numpy as np
import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow

sagemaker_session = sagemaker.Session()

role = get_execution_role()
region = sagemaker_session.boto_session.region_name


autoencoder = TensorFlow(
    entry_point="gaussian-denoising-model.py",
    role=role,
    instance_count=2,
    instance_type="ml.m5.xlarge",
    framework_version="2.1.0",
    py_version="py3",
    distribution={"parameter_server": {"enabled": True}},
)

autoencoder.fit()
