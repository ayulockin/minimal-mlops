import numpy as np
import tensorflow as tf

from medmnist import INFO


def download_and_prepare_dataset(data_info: dict):
    """
    Utility function to download the dataset and return train/valid/test images/labels.

    Arguments:
        data_info (dict): Dataset metadata
    """
    data_path = tf.keras.utils.get_file(origin=data_info['url'], md5_hash=data_info['MD5'])

    with np.load(data_path) as data:
        # Get images
        train_images = data['train_images']
        valid_images = data['val_images']
        test_images = data['test_images']

        # Get labels
        train_labels = data['train_labels'].flatten()
        valid_labels = data['val_labels'].flatten()
        test_labels = data['test_labels'].flatten()

    return train_images, train_labels, valid_images, valid_labels, test_images, test_labels