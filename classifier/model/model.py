import os
import json
import wandb
import tempfile
import numpy as np
import ml_collections

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models


# class SimpleSupervisedModel():
#     def __init__(self, args):
#         self.args = args

#     def get_backbone(self):
#         """Get backbone for the model."""
#         weights = None
#         if self.args.train_config["use_pretrained_weights"]:
#             weights = "imagenet"

#         if self.args.train_config["backbone"] == 'vgg16':
#             base_model = tf.keras.applications.VGG16(include_top=False, weights=weights)
#             base_model.trainable = True
#         elif self.args.train_config["backbone"] == 'resnet50':
#             base_model = tf.keras.applications.ResNet50(include_top=False, weights=weights)
#             base_model.trainable = True
#         else:
#             raise NotImplementedError("Not implemented for this backbone.")

#         return base_model

#     def get_model(self):
#         """Get model."""
#         # Backbone
#         base_model = self.get_backbone()

#         # Stack layers
#         inputs = layers.Input(
#             (self.args.train_config["model_img_height"],
#              self.args.train_config["model_img_width"],
#              self.args.train_config["model_img_channels"]))

#         x = base_model(inputs, training=True)
#         x = layers.GlobalAveragePooling2D()(x)
#         if self.args.train_config["post_gap_dropout"]:
#             x = layers.Dropout(self.args.train_config["dropout_rate"])(x)
#         outputs = layers.Dense(self.args.dataset_config["num_classes"], activation='softmax')(x)

#         return models.Model(inputs, outputs)


def get_backbone(args):
    """Get backbone for the model.
    
    Args:
        args (ml_collections.ConfigDict): Configuration.
    """
    weights = None
    if args.train_config["use_pretrained_weights"]:
        weights = "imagenet"

    if args.train_config["backbone"] == 'vgg16':
        base_model = tf.keras.applications.VGG16(include_top=False, weights=weights)
        base_model.trainable = True
    elif args.train_config["backbone"] == 'resnet50':
        base_model = tf.keras.applications.ResNet50(include_top=False, weights=weights)
        base_model.trainable = True
    else:
        raise NotImplementedError("Not implemented for this backbone.")

    return base_model

def get_model(args):
    """Get an image classifier with a CNN based backbone.
    
    Args:
        args (ml_collections.ConfigDict): Configuration.
    """
    # Backbone
    base_model = get_backbone(args)

    # Stack layers
    inputs = layers.Input(shape=(
        args.train_config["model_img_height"],
        args.train_config["model_img_width"],
        args.train_config["model_img_channels"]
    ))

    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    if args.train_config["post_gap_dropout"]:
        x = layers.Dropout(args.train_config["dropout_rate"])(x)
    outputs = layers.Dense(args.dataset_config["num_classes"], activation='softmax')(x)

    return models.Model(inputs, outputs)
