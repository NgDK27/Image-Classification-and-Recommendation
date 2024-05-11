import os

from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import random
import tensorflow as tf


def flip_image(image):
    # return ImageOps.mirror(image)

    # Flip the image horizontally
    flipped_image = tf.image.flip_left_right(image)
    return flipped_image


def rotate_image(image):
    angle = random.randint(-180, 180)
    return image.rotate(angle, fillcolor='white')


def shear_image(image):
    # shear_factor = random.uniform(-0.3, 0.3)
    # return image.transform(image.size, Image.AFFINE, (1, shear_factor, 0, 0, 1, 0), fillcolor='white')

    shear_factor = tf.random.uniform([], -0.3, 0.3)
    # Apply shear transformation using affine transform
    transformed_image = tf.keras.preprocessing.image.apply_affine_transform(
        image,
        shear=shear_factor,
        fill_mode='constant',
        cval=255  # Fill with white background
    )
    return transformed_image


def crop_image(image):
    original_width, original_height = image.size

    crop_width = random.randint(original_width - 50, original_width)
    crop_height = random.randint(original_height - 50, original_height)
    left = random.randint(0, original_width - crop_width)
    top = random.randint(0, original_height - crop_height)
    right = left + crop_width
    bottom = top + crop_height
    return image.crop((left, top, right, bottom))


def color_jitter_image(image):
    # # Generate random factors for color augmentation
    # # brightness_factor = random.uniform(0.5, 1.5)
    # contrast_factor = random.uniform(0.5, 1.5)
    # saturation_factor = random.uniform(0, 2)
    # hue_factor = random.uniform(-0.5, 0.5)
    #
    # modified_image = image
    #
    # # Apply hue enhancement
    # # Note: We should use the original image for hue adjustment
    # hue = ImageEnhance.Color(modified_image)
    # modified_image = hue.enhance(hue_factor)
    #
    # # Apply brightness enhancement
    # # brightness = ImageEnhance.Brightness(modified_image)
    # # modified_image = brightness.enhance(brightness_factor)
    #
    # # Apply contrast enhancement
    # # contrast = ImageEnhance.Contrast(modified_image)
    # # modified_image = contrast.enhance(contrast_factor)
    #
    # # Apply saturation enhancement
    # saturation = ImageEnhance.Color(modified_image)
    # modified_image = saturation.enhance(saturation_factor)

    # Generate random factors for color augmentation
    contrast_factor = tf.random.uniform([], 0.5, 1.5)
    saturation_factor = tf.random.uniform([], 0, 2)
    hue_factor = tf.random.uniform([], -0.5, 0.5)

    # Apply hue, saturation, and contrast adjustments
    modified_image = tf.image.adjust_hue(image, hue_factor)
    modified_image = tf.image.adjust_saturation(modified_image, saturation_factor)
    modified_image = tf.image.adjust_contrast(modified_image, contrast_factor)

    return modified_image


def shift_image(image, move_range=50):
    # # Convert the image to a NumPy array
    # image_array = np.array(image)
    #
    # # Calculate random shifts within the specified range
    # max_shift = move_range // 2
    # shift_y = random.randint(-max_shift, max_shift)
    # shift_x = random.randint(-max_shift, max_shift)
    #
    # # Calculate new positions by applying the random shifts
    # new_y_start = max(0, shift_y)
    # new_y_end = min(image_array.shape[0], image_array.shape[0] + shift_y)
    # new_x_start = max(0, shift_x)
    # new_x_end = min(image_array.shape[1], image_array.shape[1] + shift_x)
    #
    # # Create a new image array with a white background
    # shifted_image_array = np.full_like(image_array, 255)
    #
    # # Copy the shifted portion of the image to the new location within bounds
    # shifted_image_array[new_y_start:new_y_end, new_x_start:new_x_end] = image_array[max(0, -shift_y):min(image_array.shape[0], image_array.shape[0] - shift_y), max(0, -shift_x):min(image_array.shape[1], image_array.shape[1] - shift_x)]
    #
    # # Convert the NumPy array back to an image
    # shifted_image = Image.fromarray(shifted_image_array)

    # Randomly generate shift values
    shift_y = tf.random.uniform([], -move_range // 2, move_range // 2, dtype=tf.int32)
    shift_x = tf.random.uniform([], -move_range // 2, move_range // 2, dtype=tf.int32)

    # Shift the image using tf.roll
    shifted_image = tf.roll(image, shift=[shift_y, shift_x], axis=[0, 1])

    # Pad the shifted image to maintain the original size
    shifted_image = tf.pad(shifted_image, [[tf.abs(shift_y), tf.abs(shift_y)], [tf.abs(shift_x), tf.abs(shift_x)], [0, 0]])

    # Crop to the original size
    shifted_image = shifted_image[tf.abs(shift_y):tf.abs(shift_y) + image.shape[0], tf.abs(shift_x):tf.abs(shift_x) + image.shape[1], :]

    return shifted_image


def stretch_image(image):
    # # Calculate the original dimensions
    # original_width, original_height = image.size
    #
    # # Define default range for stretch factor (adjust as needed)
    # min_stretch = -0.5  # Minimum stretch factor (inward stretch)
    # max_stretch = 0.5   # Maximum stretch factor (outward stretch)
    #
    # # Generate a random stretch factor within the specified range
    # stretch_factor = random.uniform(min_stretch, max_stretch)
    #
    # # Calculate new dimensions based on the stretch factor
    # new_height = int(original_height * (1 + stretch_factor))
    #
    # # Resize the image based on the calculated dimensions
    # resized_image = image.resize((original_width, new_height), resample=Image.BICUBIC)
    #
    # # Create a new canvas
    # final_image = Image.new('RGB', (original_width, original_height), (255, 255, 255))  # White background
    #
    # if new_height <= original_height:
    #     paste_position = (0, (original_height - new_height) // 2)
    #     final_image.paste(resized_image, paste_position)
    # else:
    #     new_width = int(original_width * (original_height / new_height))
    #
    #     # Resize the image to the calculated square dimensions
    #     resized_image = resized_image.resize((new_width, original_height), resample=Image.BICUBIC)
    #
    #     # Calculate paste position to center the resized image within the canvas
    #     paste_position = ((original_width - new_width) // 2, 0)
    #     final_image.paste(resized_image, paste_position)
    #
    # return final_image

    # Get the original dimensions of the image
    original_height = tf.shape(image)[0]
    original_width = tf.shape(image)[1]

    # Define default range for stretch factor (adjust as needed)
    min_stretch = -0.5  # Minimum stretch factor (inward stretch)
    max_stretch = 0.5   # Maximum stretch factor (outward stretch)

    # Generate a random stretch factor within the specified range
    stretch_factor = tf.random.uniform([], min_stretch, max_stretch)

    # Calculate new height based on the stretch factor
    new_height = tf.cast(tf.cast(original_height, tf.float32) * (1 + stretch_factor), tf.int32)

    # Resize the image based on the calculated dimensions
    resized_image = tf.image.resize(image, (original_width, new_height), method=tf.image.ResizeMethod.BICUBIC)

    if new_height < original_height:
        # Crop the resized image to match the original height
        y_offset = (original_height - new_height) // 2
        stretched_image = resized_image[y_offset:y_offset + original_height, :, :]
    else:
        # Pad the resized image with white pixels to match the original height
        pad_top = (new_height - original_height) // 2
        pad_bottom = new_height - original_height - pad_top
        stretched_image = tf.pad(resized_image, [[pad_top, pad_bottom], [0, 0], [0, 0]], constant_values=255.0)

    return stretched_image


def augment_image(image, **kwargs):
    # Define augmentation functions with corresponding parameters
    augmentation_functions = [
        (flip_image, 'flip'),
        (stretch_image, 'stretch'),
        # (shift_image, 'shift'),
        # (rotate_image, 'rotate'),
        # (shear_image, 'shear'),
        # (crop_image, 'crop'),
        (color_jitter_image, 'color_jitter'),
    ]

    # Filter out functions based on specified kwargs
    enabled_functions = [(func, param) for func, param in augmentation_functions if kwargs.get(param, True)]

    # Randomly decide whether to apply each enabled function
    for func, param in enabled_functions:
        if random.random() < 0.5:  # Adjust probability threshold as needed (e.g., 0.5 for 50% chance)
            image = func(image)

    return image


def augment_images_from_paths(path_arr, base_path, out_path):
    for path in path_arr:
        input_path = os.path.join(base_path, path)
        filename = os.path.basename(path)
        output_path = os.path.join(out_path, filename)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        img = Image.open(input_path)
        augment_img = augment_image(img)
        augment_img.save(output_path)

