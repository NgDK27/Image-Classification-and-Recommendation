import tensorflow as tf
import matplotlib.pyplot as plt
import textwrap
from sklearn.preprocessing import LabelEncoder

from utils.augmentation import augment_image


def prepare_image_dataset(df, img_height, img_width, batch_size, base_path='../data/raw/Furniture_Data',
                          label_encoder=None):

    prepared_df = df.assign(Path=df['Path'].apply(lambda path: base_path + "/" + path))

    # Perform label encoding on the class labels
    if label_encoder is None:
        label_encoder = LabelEncoder()
        prepared_df['Class_Encoded'] = label_encoder.fit_transform(prepared_df['Class'])
    else:
        prepared_df['Class_Encoded'] = label_encoder.transform(prepared_df['Class'])

    dataset = tf.data.Dataset.from_tensor_slices(
        (prepared_df['Path'].values,
         prepared_df["Duplicate_Type"].values,
         prepared_df['Class_Encoded'].values)
    )

    image_ds = dataset.map(lambda path, duplicate_type, class_label:
                           (
                               process_image_from_path(image_path=path,
                                                       img_height=img_height,
                                                       img_width=img_width,
                                                       to_augment=duplicate_type),
                               class_label
                           ),
                           num_parallel_calls=tf.data.AUTOTUNE
                           )

    image_ds = image_ds.batch(batch_size)

    return image_ds, label_encoder


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


def process_image_from_path(image_path, img_height, img_width, to_augment):
    # Read image
    img = tf.io.read_file(image_path)

    # Decode to RGB
    img = tf.io.decode_jpeg(img, channels=3)

    # Resize
    img = tf.image.resize(img, [img_height, img_width])

    # Augment
    is_duplicate = tf.equal(to_augment, "Duplicate")

    img = tf.cond(is_duplicate, lambda: augment_image(img), lambda: img)

    rescaling_layer = tf.keras.layers.Rescaling(scale=1. / 255)
    img = rescaling_layer(img)

    return img


def one_hot_encode(image, label):
    # Normalize image data if needed
    image = image / 255.0

    # Convert label to one-hot encoded vector
    label = tf.keras.utils.to_categorical(label, num_classes=17)

    return image, label
