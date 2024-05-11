import os

import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import textwrap

from utils.augmentation import augment_image


# def create_image_data_generator(rescale=1./255, data_format='channels_last'):
#     return ImageDataGenerator(rescale=rescale, data_format=data_format)


def load_and_process_image(image_path, augment: bool, img_height, img_width):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3) # decoding

    def augment_image_tf(img):
        # Flip horizontally
        img = tf.image.flip_left_right(img)
        # Adjust brightness
        img = tf.image.random_brightness(img, max_delta=0.8)
        # Adjust contrast
        img = tf.image.random_contrast(img, lower=0.2, upper=1.8)

        return img

    if augment:
        img = augment_image_tf(img)
    img = tf.image.resize(img, [img_height, img_width]) # resizing
    return img


def prepare_image_dataset(df, labels, img_height, img_width, batch_size, base_path=''):
    """Prepare image dataset with base path inclusion, including image path in the dataset."""
    def map_fn(path, augment_flag):
        return load_and_process_image(path, augment_flag, img_height, img_width)

    paths = df["Path"].values
    full_paths = []

    # Iterate over each path and join with base_path
    for path in paths:
        full_path = os.path.join(base_path, path)
        full_paths.append(full_path)

    augment_flags = df["Augment"].values

    ds = tf.data.Dataset.from_tensor_slices((full_paths, augment_flags))
    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)

    # Zip the images dataset `ds` with the labels dataset `labels_ds`
    combined_ds = tf.data.Dataset.zip((ds, labels_ds))
    combined_ds = combined_ds.batch(batch_size)
    return combined_ds


def show_batch(image_batch, path_batch):
    """Visualize an image batch along with their paths."""
    plt.figure(figsize=(15, 15))
    for i in range(min(9, image_batch.shape[0])):  # Ensure not to exceed batch size or 9
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i])
        path_info = textwrap.fill(path_batch[i], width=50)  # Wrap text, no need to decode
        plt.xlabel(path_info, fontsize=8)  # Display the path below the image
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
    plt.tight_layout()
    plt.show()
