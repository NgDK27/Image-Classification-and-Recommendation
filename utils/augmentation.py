import os

from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import random
import tensorflow as tf


def augment_image(image, **kwargs):
    # # Define augmentation functions with corresponding parameters
    # augmentation_functions = [
    #     (flip_image, 'flip'),
    #     (stretch_image, 'stretch'),
    #     # (shift_image, 'shift'),
    #     # (rotate_image, 'rotate'),
    #     # (shear_image, 'shear'),
    #     # (crop_image, 'crop'),
    #     (color_jitter_image, 'color_jitter'),
    # ]
    #
    # # Filter out functions based on specified kwargs
    # enabled_functions = [(func, param) for func, param in augmentation_functions if kwargs.get(param, True)]
    #
    # # Randomly decide whether to apply each enabled function
    # for func, param in enabled_functions:
    #     if random.random() < 0.5:  # Adjust probability threshold as needed (e.g., 0.5 for 50% chance)
    #         image = func(image)
    #
    # return image

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_hue(image, 0.2)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    return image


# def flip_image(image):
#     # return ImageOps.mirror(image)
#
#     # Flip the image horizontally
#     flipped_image = tf.image.flip_left_right(image)
#     return flipped_image
#
#
# def rotate_image(image):
#     angle = random.randint(-180, 180)
#     return image.rotate(angle, fillcolor='white')
#
#
# def color_jitter_image(image):
#     # Generate random factors for color augmentation
#     contrast_factor = tf.random.uniform([], 0.5, 1.5)
#     saturation_factor = tf.random.uniform([], 0, 2)
#     hue_factor = tf.random.uniform([], -0.5, 0.5)
#
#     # Apply hue, saturation, and contrast adjustments
#     modified_image = tf.image.adjust_hue(image, hue_factor)
#     modified_image = tf.image.adjust_saturation(modified_image, saturation_factor)
#     modified_image = tf.image.adjust_contrast(modified_image, contrast_factor)
#
#     return modified_image
#
#
# def stretch_image(image):
#     # # Calculate the original dimensions
#     # original_width, original_height = image.size
#     #
#     # # Define default range for stretch factor (adjust as needed)
#     # min_stretch = -0.5  # Minimum stretch factor (inward stretch)
#     # max_stretch = 0.5   # Maximum stretch factor (outward stretch)
#     #
#     # # Generate a random stretch factor within the specified range
#     # stretch_factor = random.uniform(min_stretch, max_stretch)
#     #
#     # # Calculate new dimensions based on the stretch factor
#     # new_height = int(original_height * (1 + stretch_factor))
#     #
#     # # Resize the image based on the calculated dimensions
#     # resized_image = image.resize((original_width, new_height), resample=Image.BICUBIC)
#     #
#     # # Create a new canvas
#     # final_image = Image.new('RGB', (original_width, original_height), (255, 255, 255))  # White background
#     #
#     # if new_height <= original_height:
#     #     paste_position = (0, (original_height - new_height) // 2)
#     #     final_image.paste(resized_image, paste_position)
#     # else:
#     #     new_width = int(original_width * (original_height / new_height))
#     #
#     #     # Resize the image to the calculated square dimensions
#     #     resized_image = resized_image.resize((new_width, original_height), resample=Image.BICUBIC)
#     #
#     #     # Calculate paste position to center the resized image within the canvas
#     #     paste_position = ((original_width - new_width) // 2, 0)
#     #     final_image.paste(resized_image, paste_position)
#     #
#     # return final_image
#
#     # Get the original dimensions of the image
#     original_height = tf.shape(image)[0]
#     original_width = tf.shape(image)[1]
#
#     # Define default range for stretch factor (adjust as needed)
#     min_stretch = -0.5  # Minimum stretch factor (inward stretch)
#     max_stretch = 0.5   # Maximum stretch factor (outward stretch)
#
#     # Generate a random stretch factor within the specified range
#     stretch_factor = tf.random.uniform([], min_stretch, max_stretch)
#
#     # Calculate new height based on the stretch factor
#     new_height = tf.cast(tf.cast(original_height, tf.float32) * (1 + stretch_factor), tf.int32)
#
#     # Resize the image based on the calculated dimensions
#     resized_image = tf.image.resize(image, (original_width, new_height), method=tf.image.ResizeMethod.BICUBIC)
#
#     if new_height < original_height:
#         # Crop the resized image to match the original height
#         y_offset = (original_height - new_height) // 2
#         stretched_image = resized_image[y_offset:y_offset + original_height, :, :]
#     else:
#         # Pad the resized image with white pixels to match the original height
#         pad_top = (new_height - original_height) // 2
#         pad_bottom = new_height - original_height - pad_top
#         stretched_image = tf.pad(resized_image, [[pad_top, pad_bottom], [0, 0], [0, 0]], constant_values=255.0)
#
#     return stretched_image
#
#
#
#
#
# def augment_images_from_paths(path_arr, base_path, out_path):
#     for path in path_arr:
#         input_path = os.path.join(base_path, path)
#         filename = os.path.basename(path)
#         output_path = os.path.join(out_path, filename)
#
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
#         img = Image.open(input_path)
#         augment_img = augment_image(img)
#         augment_img.save(output_path)

