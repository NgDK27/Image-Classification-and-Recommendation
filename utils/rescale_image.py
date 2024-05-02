import pandas as pd
import os

from PIL import Image
from io import BytesIO


def resize_image(image_path, base_path, size=(350, 350)):
    full_path = os.path.join(base_path, image_path)
    try:
        with Image.open(full_path) as img:
            if img.size != size:
                img = img.resize(size, Image.Resampling.LANCZOS)
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format='JPEG')
                resized_data = img_byte_arr.getvalue()
            else:
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format='JPEG')
                resized_data = img_byte_arr.getvalue()
            return resized_data, img.size
    except Exception as e:
        print(f"Error resizing image {full_path}: {e}")
        return None, (0, 0)


def resize_images_in_dataframe(df, base_path, size=(350, 350)):
    results = df['Path'].apply(lambda x: resize_image(x, base_path, size))

    # Update df's numerical columns based on the resized results
    df['ResizedImage'], sizes = zip(*results)
    df['Width'], df['Height'] = zip(*sizes)
    df['Ratio'] = df['Width'] / df['Height']

    return df
