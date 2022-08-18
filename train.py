import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl
import numpy as np
import pandas as pd

import tensorflow as tf

from classifier.data import download_and_get_dataset
from classifier.model import get_model


# Download and get dataset
info, (train_images, train_labels) = download_and_get_dataset('bloodmnist', 'train')
_, (train_images, train_labels) = download_and_get_dataset('bloodmnist', 'valid')


# Get model
model = get_model()

