import numpy as np
import pandas as pd
from PIL import Image
import io
import os


def decode_and_convert_to_array(image_bytes):
    image_pil = Image.open(io.BytesIO(image_bytes))
    image_array = np.array(image_pil)
    scaled_image = image_array / 255.0
    return scaled_image


def normalize_and_save_chunk(dataframe_name, chunk_num, chunk, save_path):
    normalized_images = []
    for image_bytes in chunk['ResizedImage']:
        normalized_images.append(decode_and_convert_to_array(image_bytes))
    chunk_df = pd.DataFrame({'NormalizedImage': normalized_images})
    file_path = os.path.join(save_path, f'{dataframe_name}_normalized_images_chunk_{chunk_num}.csv')
    chunk_df.to_csv(file_path, index=False, header=False)


def process_images(dataframe_name, df, chunk_size, save_path):
    num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        chunk = df.iloc[start_idx:end_idx].copy()
        normalize_and_save_chunk(dataframe_name, i, chunk, save_path)


def insert_normalized_images(dataframe_name, df, directory):
    chunk_size = 10000
    num_chunks = len(df) // chunk_size + (0 if len(df) % chunk_size == 0 else 1)
    
    for i in range(num_chunks):
        file_path = os.path.join(directory, f'{dataframe_name}_normalized_images_chunk_{i}.csv')
        chunk_df = pd.read_csv(file_path, header=None, names=['NormalizedImage'])
        
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        
        df.iloc[start_idx:end_idx, df.columns.get_loc('NormalizedImage')] = chunk_df['NormalizedImage'].values

    return df