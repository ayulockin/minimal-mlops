import numpy as np
import tensorflow as tf

from classifier.data import INFO


def download_and_get_dataset(
    dataset_name: str='bloodmnist',
    split: str='train'
):
    """
    Utility function to download the dataset and return train/valid/test images/labels.

    Arguments:
        dataset_name (str): Name of the MedMNIST 2D dataset to be downloaded.
        split (str): Split of the dataset to be downloaded. Allowed values are `train`,
            `valid`, and `test`
    """
    info = INFO.get(dataset_name, None)
    if info is None:
        KeyError(f"The provided dataset_name: {dataset_name} is incorrect")

    data_path = tf.keras.utils.get_file(origin=info['url'], md5_hash=info['MD5'])

    with np.load(data_path) as data:
        if split=='train':
            train_images = data['train_images']
            train_labels = data['train_labels'].flatten()
            return info, (train_images, train_labels)
        elif split=='valid':
            valid_images = data['val_images']
            valid_labels = data['val_labels'].flatten()
            return info, (valid_images, valid_labels)
        elif split=='test':
            test_images = data['test_images']
            test_labels = data['test_labels'].flatten()
            return info, (test_images, test_labels)
