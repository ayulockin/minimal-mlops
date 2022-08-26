import os
import ml_collections


def get_wandb_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.project = "medmnist2D"
    configs.log_data_type = "train"
    configs.log_num_samples = -1 # passing -1 will upload the complete dataset
    configs.log_evaluation_table = False
    # configs.entity = "wandb_fc"

    return configs

def get_dataset_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.dataset_name = "bloodmnist"
    configs.image_height = 28
    configs.image_width = 28
    configs.channels = 3
    configs.apply_resize = True
    configs.batch_size = 128
    configs.num_classes = 8
    configs.apply_one_hot = True
    configs.do_cache = False

    return configs

def get_model_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.model_img_height = 28
    configs.model_img_width = 28
    configs.model_img_channels = 3
    configs.backbone = "resnet50"
    configs.use_pretrained_weights = True
    configs.dropout_rate = 0.5
    configs.post_gap_dropout = False

    return configs

def get_callback_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    # Early stopping
    configs.use_earlystopping = True
    configs.early_patience = 6
    # Reduce LR on plateau
    configs.use_reduce_lr_on_plateau = False
    configs.rlrp_factor = 0.2
    configs.rlrp_patience = 3
    # Model checkpointing
    configs.checkpoint_filepath = "wandb/model_{epoch}"
    configs.save_best_only = True
    # Model Prediction Viz
    configs.viz_num_images = 100

    return configs

def get_train_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.epochs = 3
    configs.use_augmentations = False
    configs.use_class_weights = False
    configs.optimizer = "adam"
    configs.sgd_momentum = 0.9
    configs.loss = "categorical_crossentropy"
    configs.metrics = ["accuracy"]

    return configs

# TODO (ayulockin): remove get_config to a different py file
# and condition it with config_string as referenced here:
# https://github.com/google/ml_collections#parameterising-the-get_config-function
def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.wandb_config = get_wandb_configs()
    config.dataset_config = get_dataset_configs()
    config.model_config = get_model_configs()
    config.callback_config = get_callback_configs()
    config.train_config = get_train_configs()

    return config
