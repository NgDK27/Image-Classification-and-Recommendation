import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import textwrap


def create_image_data_generator(rescale=1./255, data_format='channels_last'):
    return ImageDataGenerator(rescale=rescale, data_format=data_format)


def load_and_process_image(image_path, img_height, img_width, img_data_generator):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img = img_data_generator.standardize(img)
    return img


def prepare_image_dataset(paths, img_height, img_width, batch_size, img_data_generator, base_path=''):
    """Prepare image dataset with base path inclusion, including image path in the dataset."""
    paths = [base_path + '/' + path for path in paths]
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    image_ds = path_ds.map(lambda x: (load_and_process_image(x, img_height, img_width, img_data_generator), x),
                           num_parallel_calls=tf.data.AUTOTUNE)
    image_ds = image_ds.batch(batch_size)
    return image_ds


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