import tensorflow as tf
import matplotlib.pyplot as plt
import textwrap
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from utils.augmentation import augment_image



def process_image_from_path(image_path, img_height, img_width, to_augment="Unique", for_model=False):
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

    if for_model:
        img = tf.expand_dims(img, axis=0)

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


def prepare_image_dataset(df, img_height=256, img_width=256, batch_size=32, base_path='../data/raw/Furniture_Data'):
    # Complete path
    prepared_df = df.assign(Path=df['Path'].apply(lambda path: base_path + "/" + path))

    # Convert to tensor
    dataset = tf.data.Dataset.from_tensor_slices(prepared_df['Path'].values)

    image_ds = dataset.map(lambda path:
                           (
                               process_image_from_path(image_path=path,
                                                       img_height=img_height,
                                                       img_width=img_width),
                           ),
                           num_parallel_calls=tf.data.AUTOTUNE
                           )

    image_ds = image_ds.batch(batch_size)

    return image_ds
