import os
import ml_collections


def get_wandb_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.project = "medmnist2D"
    configs.log_validation_table = False
    # configs.entity = "wandb_fc"

    return configs

def get_dataset_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.dataset_name = "bloodmnist"
    configs.image_height = 28
    configs.image_width = 28
    configs.channels = 3
    configs.apply_resize = False
    configs.batch_size = 128
    configs.num_classes = 10
    configs.apply_one_hot = True
    configs.do_cache = False

    return configs

def get_train_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.model_img_height = 28
    configs.model_img_width = 28
    configs.model_img_channels = 3
    configs.epochs = 3
    configs.backbone = "resnet50"
    configs.use_pretrained_weights = True
    configs.use_augmentations = False
    configs.use_class_weights = False
    configs.post_gap_dropout = False
    configs.use_lr_scheduler = False
    configs.dropout_rate = 0.5
    configs.optimizer = "adam"
    configs.sgd_momentum = 0.9
    configs.loss = "categorical_crossentropy"
    configs.metrics = ["accuracy"]
    configs.early_patience = 6
    configs.rlrp_factor = 0.2
    configs.rlrp_patience = 3
    configs.resume = False

    return configs

def get_lr_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.init_lr_rate = 0.001
    configs.cosine_decay_steps = 1000
    configs.exp_decay_steps = 1000
    configs.exp_decay_rate = 0.96
    configs.exp_is_staircase = True

    return configs

# TODO (ayulockin): remove get_config to a different py file
# and condition it with config_string as referenced here:
# https://github.com/google/ml_collections#parameterising-the-get_config-function
def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.wandb_config = get_wandb_configs()
    config.dataset_config = get_dataset_configs()
    config.train_config = get_train_configs()
    config.lr_config = get_lr_configs()

    return config