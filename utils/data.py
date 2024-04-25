import os
from PIL import Image
import pandas as pd

def load(base_path: str) -> pd.DataFrame:
    """
    Load a directory of furniture images into a dataframe. 
    Directory contains 6 folders for 6 furniture classes:
        - beds
        - chairs
        - dressers
        - lamps
        - sofas
        - tables
    Each of the 6 folders contains 17 interior styles folders of that type: 
    (a) Asian; (b) Beach; (c) Contemp; (d) Craftsman; (e) Eclectic; (f) Farmhouse; (g) Industrial 
    (h) Media; (i) Midcentury; (j) Modern; (k) Rustic; (l) Scandinavian; (m) Southwestern 
    (n) Traditional; (o) Transitional; (p) Tropical and (q) Victorian 

    The resulting DataFrame has the following columns:

        - Path: Path of the image.
        - Type: Extension type of the image.
        - Width: Width (in pixels) of the image.
        - Height: Height (in pixels) of the image.
        - Ratio: Aspect ratio of the image (Width/Height).
        - Mode: Mode of the image. Define the type and depth of a pixel in the image
        - Bands: A string containing all bands of this image, separated by a space character. Read more about bands: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#bands
        - Class: Category type from 6 furniture class.
        - Style: Interior style of the furniture image.

    :param base_path: Directory of the dataset to be loaded.
    :return: A Pandas ``DataFrame``.
    """

    furniture_data = []

    for root, dirs, files in os.walk(base_path):
        for furniture_class in dirs:
            if furniture_class not in ['beds', 'chairs', 'dressers', 'lamps', 'sofas', 'tables']:
                continue
            print(f'Loading {furniture_class}...')

            for style in os.listdir(os.path.join(root, furniture_class)):
                if style == '.DS_Store':
                    continue
                print(f'Loading {furniture_class}/{style}...')

                for path in os.listdir(os.path.join(root, furniture_class, style)):
                    full_path = os.path.join(root, furniture_class, style, path)
                    if os.path.isdir(full_path):
                        continue

                    with Image.open(os.path.join(root, furniture_class, style, path)) as im:
                        furniture_data.append({
                            'Path': f'{furniture_class}/{style}/{path}',
                            'Type': path.split('.')[-1],
                            'Width': im.size[0],
                            'Height': im.size[1],
                            'Ratio': im.size[0]/im.size[1],
                            'Mode': im.mode,
                            'Class': furniture_class,
                            'Style': style
                        })
    
    return pd.DataFrame(furniture_data)
