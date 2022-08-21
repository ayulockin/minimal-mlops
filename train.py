import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

import wandb
import tensorflow as tf

from classifier.data import download_and_get_dataset
from classifier.data import GetDataloader
from classifier.model import get_model

# Config
FLAGS = flags.FLAGS
flags.DEFINE_bool("wandb", False, "MLOps pipeline for our classifier.")
CONFIG = config_flags.DEFINE_config_file("config")


def main(_):
    # Get configs from the config file.
    config = CONFIG.value
    print(config)

    if FLAGS.wandb:
        run = wandb.init(
            project=CONFIG.value.wandb_config.project,
            job_type='train',
            config=config.to_dict(),
        )

    # Download and get dataset
    dataset_name = config.dataset_config.dataset_name
    info, (train_images, train_labels) = download_and_get_dataset(dataset_name, 'train')
    info, (valid_images, valid_labels) = download_and_get_dataset(dataset_name, 'valid')

    # Get dataloader
    make_dataloader = GetDataloader(config)
    trainloader = make_dataloader.get_dataloader(train_images, train_labels)
    validloader = make_dataloader.get_dataloader(valid_images, valid_labels)

    # Get model
    tf.keras.backend.clear_session()
    model = get_model(config)
    model.summary()

    # Compile the model
    model.compile(
        optimizer = config.train_config.optimizer,
        loss = config.train_config.loss,
        metrics = config.train_config.metrics
    )

    # Train the model
    model.fit(
        trainloader,
        validation_data = validloader,
        epochs = config.train_config.epochs
    )

if __name__ == "__main__":
    app.run(main)
