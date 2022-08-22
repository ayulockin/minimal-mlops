import os
import wandb
import tensorflow as tf


def get_earlystopper(args):
    args = args.callback_config

    earlystopper = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=args.early_patience, verbose=0, mode='auto',
        restore_best_weights=True
    )

    return earlystopper

def get_reduce_lr_on_plateau(args):
    args = args.callback_config

    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=args.rlrp_factor,
        patience=args.rlrp_patience
    )

    return reduce_lr_on_plateau


class WandbModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self,
                 filepath='wandb/model_{epoch}',
                 monitor='val_loss',
                 save_best_only=False,
                 save_weights_only=False,
                 initial_value_threshold=None,
                 **kwargs):
        super(WandbModelCheckpoint, self).__init__(
            filepath,
            monitor,
            save_best_only,
            save_weights_only,
            initial_value_threshold,
            **kwargs
        )

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        # Get filepath where the model checkpoint is saved.
        filepath = self._get_file_path(epoch, batch=None, logs=logs)
        # Log the model as artifact
        self._log_ckpt_as_artifact(filepath)

    def _log_ckpt_as_artifact(self, filepath):
        """Log model checkpoint as  W&B Artifact."""
        model_artifact = wandb.Artifact(
            f'run_{wandb.run.id}_model', type='model')
        model_artifact.add_dir(filepath)
        wandb.log_artifact(model_artifact)


def get_model_checkpoint_callback(args):
    model_checkpointer = WandbModelCheckpoint(
        filepath='wandb/model_{epoch}',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        initial_value_threshold=None,
    )

    return model_checkpointer

