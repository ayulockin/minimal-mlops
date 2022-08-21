import numpy as np
import tensorflow as tf
from functools import partial
import albumentations as A

AUTOTUNE = tf.data.AUTOTUNE

class GetDataloader():
    def __init__(self, args):
        self.args = args

    def get_dataloader(self, images, labels, dataloader_type='train'):
        '''
        Args:
            images: List of images loaded in the memory.
            labels: List of one labels.
            dataloader_type: The type of the dataloader, can be `train`,
                `valid`, or `test`.

        Return:
            dataloader: train, validation or test dataloader
        '''
        # Consume dataframe
        dataloader = tf.data.Dataset.from_tensor_slices((images, labels))

        # Shuffle if its for training
        if dataloader_type=='train':
            dataloader = dataloader.shuffle(self.args.dataset_config.batch_size)

        # Load the image
        dataloader = (
            dataloader
            .map(partial(self.parse_data, dataloader_type=dataloader_type), num_parallel_calls=AUTOTUNE)
        )

        if self.args.dataset_config.do_cache:
            dataloader = dataloader.cache()

        # Add general stuff
        dataloader = (
            dataloader
            .batch(self.args.dataset_config.batch_size)
            .prefetch(AUTOTUNE)
        )

        return dataloader

    def preprocess_image(self, img, dataloader_type='train'):
        # Scale image
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        # resize the image to the desired size
        if self.args.dataset_config.apply_resize:
            if dataloader_type=='train':
                img = tf.image.resize(
                    img,
                    [self.args.dataset_config.image_height, self.args.dataset_config.image_width]
                )
            elif dataloader_type=='valid' or dataloader_type=='test':
                img = tf.image.resize(
                    img, 
                    [self.args.train_config.model_img_height, self.args.train_config.model_img_width],
                )
            else:
                raise NotImplementedError("No data type")

            img = tf.clip_by_value(img, 0.0, 1.0)

        return img

    def parse_data(self, image, label, dataloader_type='train'):
        # Parse Image
        image = self.preprocess_image(image)

        if dataloader_type in ['train', 'valid']:
            # Parse Target
            label = tf.cast(label, dtype=tf.int64)
            if self.args.dataset_config.apply_one_hot:
                label = tf.one_hot(
                    label,
                    depth=self.args.dataset_config.num_classes
                )
            return image, label
        elif dataloader_type == 'test':
            return image
        else:
            raise NotImplementedError("Not implemented for this data_type")
