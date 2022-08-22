import wandb
import tensorflow as tf
from classifier.utils import WandbTablesBuilder


class WandbClfEvalCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 dataloader,
                 num_samples=100):
        super(WandbClfEvalCallback, self).__init__()
        self.dataloader = dataloader.unbatch().take(num_samples)
        self.tables_builder = WandbTablesBuilder()

    def on_train_begin(self, logs=None):
        self.tables_builder.init_data_table(
            column_names = ["image_index", "images", "ground_truth"]
        )
        # Add validation data to the table
        self.add_ground_truth()
        # Log the table to W&B
        self.tables_builder.log_data_table()

    def on_epoch_end(self, epoch, logs=None):
        # Initialize a prediction wandb table
        self.tables_builder.init_pred_table(
            column_names = ["epoch", "image_index",
                            "images", "ground_truth", 
                            "prediction"]
        )
        # Add prediction to the table
        self.add_model_predictions(epoch)
        # Log the eval table to W&B
        self.tables_builder.log_pred_table()

    def add_ground_truth(self):
        """Logic for adding validation/training data to `data_table`.
        This method is called once `on_train_begin` or equivalent hook.
        """
        # Iterate through the samples and log them to the data_table.
        for idx, (image, label) in enumerate(self.dataloader.as_numpy_iterator()):
            # Log a row to the data table.
            self.tables_builder.data_table.add_data(
                idx,
                wandb.Image(image),
                tf.argmax(label, axis=0).numpy()
            )
    
    def add_model_predictions(self, epoch):
        # Get predicted detections
        predictions = self._infer()

        # Iterate through the samples.
        data_table_ref = self.tables_builder.data_table_ref
        table_idxs = data_table_ref.get_index()
        assert len(table_idxs) == len(predictions)

        for idx in table_idxs:
            pred = predictions[idx]

            # Log a row to the eval table.
            self.tables_builder.pred_table.add_data(
                epoch,
                data_table_ref.data[idx][0],
                data_table_ref.data[idx][1],
                data_table_ref.data[idx][2],
                pred
            )

    def _infer(self):
        predictions = []

        for idx, (image, label) in enumerate(self.dataloader.as_numpy_iterator()):
            # Get model prediction.
            pred = self.model(tf.expand_dims(image, axis=0))
            pred = tf.argmax(tf.squeeze(pred, axis=0))
            predictions.append(pred)

        return predictions

def get_evaluation_callback(args, dataloader):
    return WandbClfEvalCallback(
        dataloader,
        num_samples=100
    )
