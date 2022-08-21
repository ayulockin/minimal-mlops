import wandb
from tqdm import tqdm


def log_data_as_table(info, images, labels, data_type="train", num_samples=100):
    label_names = info["label"]
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
