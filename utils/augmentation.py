import os

from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import random


def flip_image(image):
    return ImageOps.mirror(image)


def rotate_image(image):
    angle = random.randint(-180, 180)
    return image.rotate(angle)


def shear_image(image):
    shear_factor = random.uniform(-0.5, 0.5)
    return image.transform(image.size, Image.AFFINE, (1, shear_factor, 0, 0, 1, 0))


def crop_image(image):
    crop_size = random.randint(250, 350)
    left = random.randint(0, image.width - crop_size)
    top = random.randint(0, image.height - crop_size)
    right = left + crop_size
    bottom = top + crop_size
    return image.crop((left, top, right, bottom))


def color_jitter_image(image):
    # Generate random factors for color augmentation
    # brightness_factor = random.uniform(0.5, 1.5)
    contrast_factor = random.uniform(0.5, 1.5)
    saturation_factor = random.uniform(0, 2)
    hue_factor = random.uniform(-0.5, 0.5)

    modified_image = image

    # Apply hue enhancement
    # Note: We should use the original image for hue adjustment
    hue = ImageEnhance.Color(modified_image)
    modified_image = hue.enhance(hue_factor)

    # Apply brightness enhancement
    # brightness = ImageEnhance.Brightness(modified_image)
    # modified_image = brightness.enhance(brightness_factor)

    # Apply contrast enhancement
    contrast = ImageEnhance.Contrast(modified_image)
    modified_image = contrast.enhance(contrast_factor)

    # Apply saturation enhancement
    saturation = ImageEnhance.Color(modified_image)
    modified_image = saturation.enhance(saturation_factor)

    return modified_image


def shift_image(image, move_range=50):
    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Calculate random shifts within the specified range
    max_shift = move_range // 2
    shift_y = random.randint(-max_shift, max_shift)
    shift_x = random.randint(-max_shift, max_shift)

    # Calculate new positions by applying the random shifts
    new_y_start = max(0, shift_y)
    new_y_end = min(image_array.shape[0], image_array.shape[0] + shift_y)
    new_x_start = max(0, shift_x)
    new_x_end = min(image_array.shape[1], image_array.shape[1] + shift_x)

    # Create a new image array with a white background
    shifted_image_array = np.full_like(image_array, 255)

    # Copy the shifted portion of the image to the new location within bounds
    shifted_image_array[new_y_start:new_y_end, new_x_start:new_x_end] = image_array[max(0, -shift_y):min(image_array.shape[0], image_array.shape[0] - shift_y), max(0, -shift_x):min(image_array.shape[1], image_array.shape[1] - shift_x)]

    # Convert the NumPy array back to an image
    shifted_image = Image.fromarray(shifted_image_array)

    return shifted_image


def augment_image(
        image_path,
        flip=False,
        rotate=False,
        shear=False,
        crop=False,
        color_jitter=False,
        shift=False):

    # Open the input image
    image = Image.open(image_path)

    # Augmentation techniques
    augmentation_techniques = []
    if flip:
        augmentation_techniques.append(flip_image)
    if shift:
        augmentation_techniques.append(shift_image)
    if rotate:
        augmentation_techniques.append(rotate_image)
    if crop:
        augmentation_techniques.append(crop_image)
    if shear:
        augmentation_techniques.append(shear_image)

    # Apply the selected augmentation techniques
    augmented_image = image
    for func in augmentation_techniques:
        augmented_image = func(augmented_image)

    # Convert the sheared image to a NumPy array
    image_array = np.array(augmented_image)

    # Identify black pixels (where RGB values sum up to zero)
    black_pixels = np.sum(image_array, axis=2) == 0

    # Replace black pixels with white (255)
    image_array[black_pixels] = 255

    # Convert the modified array back to an image
    augmented_image = Image.fromarray(image_array)

    if color_jitter:
        augmented_image = color_jitter_image(augmented_image)

    return augmented_image


def generate_augmented_images(
        image_path,
        num_images,
        flip=False,
        rotate=False,
        shear=False,
        crop=False,
        color_jitter=False,
        shift=False):

    # Get the directory and filename
    directory, filename = os.path.split(image_path)
    filename_without_ext, ext = os.path.splitext(filename)

    # Create a list of available numbers for filenames
    available_numbers = list(range(1, num_images + 1))
    random.shuffle(available_numbers)

    # Generate and save the augmented images
    for i in range(num_images):
        # Generate a unique filename
        new_filename = f"{filename_without_ext}_{available_numbers.pop()}{ext}"
        new_image_path = os.path.join(directory, new_filename)

        # Generate an augmented image
        augmented_image = augment_image(image_path, flip=flip, rotate=rotate, shear=shear, crop=crop, color_jitter=color_jitter, shift=shift)

        # Save the augmented image
        augmented_image.save(new_image_path)
        print(f"Saved augmented image: {new_image_path}")

