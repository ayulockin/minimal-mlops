import wandb
import tensorflow as tf


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
        if wandb.run is None:
            raise wandb.Error(
                "You must call wandb.init() before WandbModelCheckpoint()"
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
        filepath=args.callback_config.checkpoint_filepath,
        monitor='val_loss',
        save_best_only=args.callback_config.save_best_only,
        save_weights_only=False,
        initial_value_threshold=None,
    )

    return model_checkpointer