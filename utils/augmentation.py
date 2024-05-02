from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import random


def flip_image(image):
    return ImageOps.mirror(image)


def rotate_image(image):
    angle = random.randint(-180, 180)
    return image.rotate(angle, fillcolor='white')


def shear_image(image):
    shear_factor = random.uniform(-0.5, 0.5)
    return image.transform(image.size, Image.AFFINE, (1, shear_factor, 0, 0, 1, 0), fillcolor='white')


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


def stretch_image(image):
    # Calculate the original dimensions
    original_width, original_height = image.size

    # Define default range for stretch factor (adjust as needed)
    min_stretch = -0.3  # Minimum stretch factor (inward stretch)
    max_stretch = 0.3   # Maximum stretch factor (outward stretch)

    # Generate a random stretch factor within the specified range
    stretch_factor = random.uniform(min_stretch, max_stretch)

    # Calculate new dimensions based on the stretch factor
    new_height = int(original_height * (1 + stretch_factor))

    # Resize the image based on the calculated dimensions
    resized_image = image.resize((original_width, new_height), resample=Image.BICUBIC)

    # Create a new canvas of size 350x350
    final_canvas_size = (350, 350)
    final_image = Image.new('RGB', final_canvas_size, (255, 255, 255))  # White background

    if new_height <= original_height:
        # Apply padding to fit the output height to 350 (if result is smaller)
        paste_position = ((final_canvas_size[0] - original_width) // 2, 0)
        final_image.paste(resized_image, paste_position)
    else:
        # Apply padding to make the image square and then resize to 350x350 (if result is larger)
        # Calculate width to make the image square (same as canvas width)
        new_width = int(original_width * (final_canvas_size[1] / new_height))

        # Resize the image to the calculated square dimensions
        resized_image = resized_image.resize((new_width, final_canvas_size[1]), resample=Image.BICUBIC)

        # Calculate paste position to center the resized image within the canvas
        paste_position = ((final_canvas_size[0] - new_width) // 2, 0)
        final_image.paste(resized_image, paste_position)

    return final_image


def augment_image(image, **kwargs):
    # Define augmentation functions with corresponding parameters
    augmentation_functions = [
        (flip_image, 'flip'),
        (stretch_image, 'stretch'),
        (shift_image, 'shift'),
        # (rotate_image, 'rotate'),
        (shear_image, 'shear'),
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



