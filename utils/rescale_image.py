import os
import io
import random
import logging

from io import BytesIO
from PIL import Image


def resize_image(image_path, base_path, size=(200, 200)):
    full_path = os.path.join(base_path, image_path)
    try:
        with Image.open(full_path) as img:
            if img.size != size:
                img = img.resize(size, Image.Resampling.LANCZOS)
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='JPEG')
            resized_data = img_byte_arr.getvalue()
            return resized_data
    except Exception as e:
        logging.error(f"Error resizing image {full_path}: {e}")
        return None


def resize_images_in_dataframe(df, base_path, size=(200, 200)):
    df['ResizedImage'] = df['Path'].apply(lambda x: resize_image(x, base_path, size))
    return df


def load_random_images(df, num_images=3):
    random_indices = random.sample(range(len(df)), num_images)
    random_images = df.iloc[random_indices]
    return random_images


def display_random_images(df, num_images=3):
    random_images = load_random_images(df, num_images)
    
    for _, row in random_images.iterrows():
        image_data = row['ResizedImage']
        image_path = row['Path']
        
        print(f"Image Path: {image_path}")
        
        image = Image.open(io.BytesIO(image_data))
        display(image)