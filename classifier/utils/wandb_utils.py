import wandb
from tqdm import tqdm


def log_data_as_table(info, images, labels, data_type="train", num_samples=100):
    label_names = info["label"]
    if num_samples == -1:
        num_images = len(images)
    # This is the artifact to store the W&B Table.
    # We could have directly uploaded the images and `.csv` files to an artifact,
    # at the cost of no visualization. Note however that for few data formats like those
    # used in medical imaging, uploading the dataset as artifact is recommended.
    data_artifact = wandb.Artifact(name=data_type, type="dataset")
    # This is the table where images, labels and associated metadata will be logged.
    data_at = wandb.Table(columns=["image_idx", "image", "label", "encoded_label"])

    for idx, (img, label) in tqdm(enumerate(zip(images, labels))):
        # This is where we are adding the data to the table row wise.
        data_at.add_data(
            idx,
            wandb.Image(img, mode="RGB"),
            label_names[str(label)],
            int(label)
        )
        if idx+1 == num_samples:
            break

    # Store the table as artifact.
    data_artifact.add(data_at, f"{data_type}-table")
    # Now we will log the artifact to W&B.
    wandb.log_artifact(data_artifact)


class WandbTablesBuilder:
    """
    Utility class that contains useful methods to create W&B Tables,
    and log it to W&B.

    This table is particularly useful for creating evaluation table.
    """
    def init_data_table(self, column_names: list):
        """Initialize the W&B Tables for validation data.
        Call this method `on_train_begin` or equivalent hook. This is followed by
        adding data to the table row or column wise.
        Args:
            column_names (list): Column names for W&B Tables.
        """
        self.data_table = wandb.Table(columns=column_names, allow_mixed_types=True)

    def init_pred_table(self, column_names: list):
        """Initialize the W&B Tables for model evaluation.
        Call this method `on_epoch_end` or equivalent hook. This is followed by
        adding data to the table row or column wise.
        Args:
            column_names (list): Column names for W&B Tables.
        """
        self.pred_table = wandb.Table(columns=column_names)

    def log_data_table(self, 
                    name: str='val',
                    type: str='dataset',
                    table_name: str='val_data'):
        """Log the `data_table` as W&B artifact and call
        `use_artifact` on it so that the evaluation table can use the reference
        of already uploaded data (images, text, scalar, etc.).
        This allows the data to be uploaded just once.
        Args:
            name (str):  A human-readable name for this artifact, which is how 
                you can identify this artifact in the UI or reference 
                it in use_artifact calls. (default is 'val')
            type (str): The type of the artifact, which is used to organize and
                differentiate artifacts. (default is 'val_data')
            table_name (str): The name of the table as will be displayed in the UI.
        """
        data_artifact = wandb.Artifact(name, type=type)
        data_artifact.add(self.data_table, table_name)

        # Calling `use_artifact` uploads the data to W&B.
        wandb.run.use_artifact(data_artifact)
        data_artifact.wait()

        # We get the reference table.
        self.data_table_ref = data_artifact.get(table_name)

    def log_pred_table(self,
                    type: str='evaluation',
                    table_name: str='eval_data'):
        """Log the W&B Tables for model evaluation.
        The table will be logged multiple times creating new version. Use this
        to compare models at different intervals interactively.
        Args:
            type (str): The type of the artifact, which is used to organize and
                differentiate artifacts. (default is 'val_data')
            table_name (str): The name of the table as will be displayed in the UI.
        """
        pred_artifact = wandb.Artifact(
            f'run_{wandb.run.id}_pred', type=type)
        pred_artifact.add(self.pred_table, table_name)
        # TODO: Add aliases
        wandb.run.log_artifact(pred_artifact)
