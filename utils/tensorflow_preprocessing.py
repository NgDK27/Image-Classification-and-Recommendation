import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import textwrap

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from utils.augmentation import augment_image


# def load_and_process_image(image_path, img_height, img_width, img_data_generator):
#     img = tf.io.read_file(image_path)
#     img = tf.io.decode_jpeg(img, channels=3)  # decoding
#     img = tf.image.resize(img, [img_height, img_width])  # resizing
#     img = tf.keras.layers.Rescaling(img)  # normalization
#     return img
#
#
# def prepare_image_dataset(paths, img_height, img_width, batch_size, img_data_generator, base_path=''):
#     """Prepare image dataset with base path inclusion, including image path in the dataset."""
#     paths = [base_path + '/' + path for path in paths]
#     path_ds = tf.data.Dataset.from_tensor_slices(paths)
#     image_ds = path_ds.map(lambda x: (load_and_process_image(x, img_height, img_width, img_data_generator), x),
#                            num_parallel_calls=tf.data.AUTOTUNE)
#     image_ds = image_ds.batch(batch_size)
#     return image_ds


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


def process_image_from_path(image_path, img_height, img_width, to_augment="Unique"):
    # Read image
    img = tf.io.read_file(image_path)

    # Decode to RGB
    img = tf.io.decode_jpeg(img, channels=3)

    # Resize
    img = tf.image.resize(img, [img_height, img_width])

    # Augment only images marked as "Duplicate"
    is_duplicate = tf.equal(to_augment, "Duplicate")

    img = tf.cond(is_duplicate, lambda: augment_image(img), lambda: img)

    # Rescale pixel value
    rescaling_layer = tf.keras.layers.Rescaling(scale=1. / 255)
    img = rescaling_layer(img)

    return img


def prepare_image_target_dataset(df, target_name, img_height=256, img_width=256, batch_size=32,
                                 base_path='../data/raw/Furniture_Data',
                                 label_encoder=None):
    # Complete path
    prepared_df = df.assign(Path=df['Path'].apply(lambda path: base_path + "/" + path))

    # Label encoding
    new_label_encoder = label_encoder
    dataset = None

    # For "Class" target
    if target_name == "Class":
        if new_label_encoder is None:
            new_label_encoder = LabelEncoder()
            prepared_df['Target'] = new_label_encoder.fit_transform(prepared_df['Class'])
        else:
            prepared_df['Target'] = new_label_encoder.transform(prepared_df['Class'])

        # Convert to tensor
        dataset = tf.data.Dataset.from_tensor_slices(
            (prepared_df['Path'].values,
             prepared_df["Duplicate_Type"].values,
             prepared_df['Target'].values)
        )

    # For "Style" target
    elif target_name == "Style":
        if new_label_encoder is None:
            new_label_encoder = OneHotEncoder()
            labels = new_label_encoder.fit_transform(prepared_df['Style'].values.reshape(-1, 1))
            labels_df = pd.DataFrame(labels.toarray(), columns=new_label_encoder.get_feature_names_out(['Style']))
            prepared_df = pd.concat([prepared_df, labels_df], axis=1)
        else:
            labels = new_label_encoder.transform(prepared_df['Style'].values.reshape(-1, 1))
            labels_df = pd.DataFrame(labels.toarray(), columns=new_label_encoder.get_feature_names_out(['Style']))
            prepared_df = pd.concat([prepared_df, labels_df], axis=1)

        # Convert to tensor
        target_columns = new_label_encoder.get_feature_names_out(['Style'])
        dataset = tf.data.Dataset.from_tensor_slices(
            (prepared_df['Path'].values,
             prepared_df["Duplicate_Type"].values,
             prepared_df[target_columns].values)
        )

    image_ds = dataset.map(lambda path, duplicate_type, target:
                           (
                               process_image_from_path(image_path=path,
                                                       img_height=img_height,
                                                       img_width=img_width,
                                                       to_augment=duplicate_type),
                               target
                           ),
                           num_parallel_calls=tf.data.AUTOTUNE
                           )

    image_ds = image_ds.batch(batch_size)

    return image_ds, new_label_encoder
