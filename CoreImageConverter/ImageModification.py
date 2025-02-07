import os.path
from functools import singledispatch
from PIL import Image
import numpy as np
import math
import time
import threading

def open_image (image_name: str) -> np.ndarray:

    path_parts = os.path.abspath(os.getcwd()).split('\\')
    path_to_image = ''

    for part in range(len(path_parts) - 1):
        path_to_image += path_parts[part] + '\\'

    path_to_image += "ServerApp\\images_upload\\" + image_name

    if is_greyscale(path_to_image):
        image = np.array(Image.open(path_to_image))
        return convert_rgb_to_greyscale(image)
    return np.array(Image.open(path_to_image))

@singledispatch
def is_greyscale (imag):
    raise NotImplementedError("Not implemented")

@is_greyscale.register(np.ndarray)
def _(imag)->bool:
    if len(imag.shape) <= 2:
        print("Grayscale image")
        return True
    print("Color image")
    return False

@is_greyscale.register(str)
def _(imag)->bool:
    image = np.array(Image.open(imag))
    image_total_pixel_diff = 0
    sample_crop_from = [int(image.shape[0] / 2 - image.shape[0] / 20), int(image.shape[1] / 2 - image.shape[1] / 20)]
    sample_crop_to = [int(image.shape[0] / 2 + image.shape[0] / 20), int(image.shape[1] / 2 + image.shape[1] / 20)]
    sample = crop_image(image, sample_crop_from, sample_crop_to)
    for row in sample:
        for pixel in row:
            (r, g, b) = pixel
            rg = abs(r/255 - g/255)
            rb = abs(r/255 - b/255)
            gb = abs(g/255 - b/255)
            image_total_pixel_diff += rg + rb + gb

    if image_total_pixel_diff > 5:
        print("Color image")
        return False
    print("Grayscale image")
    return True

def convert_rgb_to_greyscale(image: np.ndarray) -> np.ndarray:
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

def crop_image (image: np.ndarray, point_a: [int, int], point_b: [int, int]) -> np.ndarray:
    if (point_b[0] < 0 and point_b[1] < 0) or (point_a[0] > image.shape[0] and point_a[1] > image.shape[1]):
        print("Crop size can't be less than zero")
        return image
    if point_a[0] < 0:
        point_a = 0
    if point_a[1] < 0:
        point_a = 0
    if point_b[0] > image.shape[0]:
        point_b[0] = image.shape[0]
    if point_b[1] > image.shape[1]:
        point_b[1] = image.shape[1]
    cropped_image = image[point_a[0]:point_b[0], point_a[1]:point_b[1]]

    return cropped_image

def rotate_image_right_90_deg (image: np.ndarray) -> np.ndarray:
    return np.rot90(image)

def flip_image (image: np.ndarray) -> np.ndarray:
    return np.fliplr(image)

def convert_to_negative (image: np.ndarray) -> np.ndarray:
    return 255 - image

def solarize_image (image: np.ndarray, thresh_val: int = 130, effect : int = 255) -> np.ndarray:
    if not is_greyscale(image):
        red_image, green_image, blue_image = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        red_solarization = np.where(red_image < thresh_val, red_image, effect - red_image)
        green_solarization = np.where(green_image < thresh_val, green_image, effect - green_image)
        blue_solarization = np.where(blue_image < thresh_val, blue_image, effect - blue_image)
        image_solarization = np.dstack((red_solarization, green_solarization, blue_solarization))
    else:
        image_solarization = np.where(image < thresh_val, image, effect - image)

    return image_solarization

def change_brightness (image: np.ndarray, value: int = 0) -> np.ndarray:
    image_br = image.astype(np.float64)
    image_br += value
    image_br = np.clip(image_br, 0, 255)
    return image_br.astype(np.uint8)

def change_contrast (image: np.ndarray, value: float = 0) -> np.ndarray:
    #need to lookup better ways to change contrast
    if value > 49.9:
        print("Value cannot be bigger than 49.9")
        return image
    image_min = np.percentile(image, value)
    image_max = np.percentile(image, 100-value)
    image_changed = ((np.clip(image, image_min, image_max) - image_min) / (image_max - image_min)) * 255

    return image_changed.astype(np.uint8)

def resize_image (image: np.ndarray, new_height: int | float, new_width: int | float) -> np.ndarray:
    grayscale = is_greyscale(image)
    new_width = int(new_width)
    new_height = int(new_height)
    if not grayscale:
        new_image = np.zeros((math.floor(new_height), math.floor(new_width), image.shape[2]), dtype=np.uint8)
        print(image.shape)
    else:
        new_image = np.zeros((math.floor(new_height), math.floor(new_width)), dtype=np.uint8)

    image_scale_factor_y = image.shape[0] / new_height
    image_scale_factor_x = image.shape[1] / new_width
    image_scale_factor = (image_scale_factor_y, image_scale_factor_x)

    time_start = time.time()

    for new_y in range(new_height):
        if grayscale:
            tr = threading.Thread(target=set_nearby_pixels_greyscale,
                                  args=(image, new_image, new_width, image_scale_factor, new_y))
            tr.start()
            tr.join()
        else:
            tr = threading.Thread(target=set_nearby_pixels_color,
                                  args=(image, new_image, new_width, image_scale_factor, new_y))
            tr.start()
            tr.join()

    time_end = time.time()
    print("Time elapsed: ", time_end - time_start)

    return new_image

def set_nearby_pixels_color(source_image: np.ndarray, result_image: np.ndarray, new_width: int,
                                image_scale_factor: (int, int), curr_y: int):
    for new_x in range(new_width):
        old_pixel = [curr_y * image_scale_factor[0], new_x * image_scale_factor[1]]
        pixel_fraction = [old_pixel[0] - math.floor(old_pixel[0]), old_pixel[1] - math.floor(old_pixel[1])]

        left_upper_pixel = source_image[math.floor(old_pixel[0]), math.floor(old_pixel[1])]
        right_upper_pixel = source_image[
                math.floor(old_pixel[0]), min(source_image.shape[1] - 1, math.ceil(old_pixel[1]))]
        left_lower_pixel = source_image[
                min(source_image.shape[0] - 1, math.ceil(old_pixel[0])), math.floor(old_pixel[1])]
        right_lower_pixel = source_image[
                min(source_image.shape[0] - 1, math.ceil(old_pixel[0])), min(source_image.shape[1] - 1,
                                                                             math.ceil(old_pixel[1]))]
        mix_top_pixels = left_upper_pixel * pixel_fraction[1] + right_upper_pixel * (1 - pixel_fraction[1])
        mix_bot_pixels = left_lower_pixel * pixel_fraction[1] + right_lower_pixel * (1 - pixel_fraction[1])
        mix_top_and_bot = mix_top_pixels * pixel_fraction[0] + mix_bot_pixels * (1 - pixel_fraction[0])

        result_image[curr_y, new_x] = mix_top_and_bot

def set_nearby_pixels_greyscale(source_image: np.ndarray, result_image: np.ndarray, new_width: int,
                                    image_scale_factor: (int, int), curr_y: int):
    for new_x in range(new_width):
        for channel in range(source_image.shape[2]):
            old_pixel = [curr_y * image_scale_factor[0], new_x * image_scale_factor[1]]
            pixel_fraction = [old_pixel[0] - math.floor(old_pixel[0]), old_pixel[1] - math.floor(old_pixel[1])]

            left_upper_pixel = source_image[math.floor(old_pixel[0]), math.floor(old_pixel[1]), channel]
            right_upper_pixel = source_image[
                    math.floor(old_pixel[0]), min(source_image.shape[1] - 1, math.ceil(old_pixel[1])), channel]
            left_lower_pixel = source_image[min(source_image.shape[0] - 1, math.ceil(old_pixel[0])), math.floor(old_pixel[1]), channel]
            right_lower_pixel = source_image[min(source_image.shape[0] - 1, math.ceil(old_pixel[0])), min(source_image.shape[1] - 1,
                                                                                 math.ceil(old_pixel[1])), channel]

            mix_top_pixels = left_upper_pixel * pixel_fraction[1] + right_upper_pixel * (1 - pixel_fraction[1])
            mix_bot_pixels = left_lower_pixel * pixel_fraction[1] + right_lower_pixel * (1 - pixel_fraction[1])
            mix_top_and_bot = mix_top_pixels * pixel_fraction[0] + mix_bot_pixels * (1 - pixel_fraction[0])

            result_image[curr_y, new_x, channel] = mix_top_and_bot



