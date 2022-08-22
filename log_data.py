import os
import wandb
import numpy as np
from absl import flags, app
from ml_collections.config_flags import config_flags

from classifier.data import download_and_get_dataset
from classifier.utils import log_data_as_table

# Config
FLAGS = flags.FLAGS
flags.DEFINE_bool("wandb", True, "MLOps pipeline for our classifier.")
CONFIG = config_flags.DEFINE_config_file("config")


def main(_):
    # Get configs from the config file.
    config = CONFIG.value

    # Initialize a Weights and Biases run.
    if FLAGS.wandb:
        run = wandb.init(
            project=CONFIG.value.wandb_config.project,
            job_type='train',
            config=config.to_dict(),
        )

    data_type = config.wandb_config.log_data_type

    # Download and get dataset
    dataset_name = config.dataset_config.dataset_name
    info, (images, labels) = download_and_get_dataset(dataset_name, data_type)

    # Log the dataset as W&B Tables
    log_data_as_table(
        info,
        images,
        labels,
        data_type=data_type,
        num_samples=config.wandb_config.log_num_samples
    )

if __name__ == "__main__":
    app.run(main)
