
from functools import singledispatch
from PIL import Image
import numpy as np



def open_image (address: str) -> np.ndarray:
    if is_grayscale(address):
        image = np.array(Image.open(address))
        print(image.shape)
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return np.array(Image.open(address))

@singledispatch
def is_grayscale (img):
    raise NotImplementedError("Not implemented")

@is_grayscale.register(np.ndarray)
def _(img)->bool:
    return True

@is_grayscale.register(str)
def _(img)->bool:
    image = np.array(Image.open(img))
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
    image_negative = 255 - image
    img = Image.fromarray(image_negative)
    img.show()
    return image_negative

def solarize_image (image: np.ndarray, thresh_val: int = 130, effect : int = 255) -> np.ndarray:
    if not is_grayscale(image):
        red_image, green_image, blue_image = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        red_solarization = np.where(red_image < thresh_val, red_image, effect - red_image)
        green_solarization = np.where(green_image < thresh_val, green_image, effect - green_image)
        blue_solarization = np.where(blue_image < thresh_val, blue_image, effect - blue_image)

        image_solarization = np.dstack((red_solarization, green_solarization, blue_solarization))

    else:
        image_solarization = np.where(image < thresh_val, image, effect - image)

    return image_solarization

def change_brightness (image: np.ndarray, value: int) -> np.ndarray:
    image_br = image.astype(np.float64)
    image_br += value
    image_br = np.clip(image_br, 0, 255)
    return image_br.astype(np.uint8)

def change_contrast (image: np.ndarray, value: float = 0) -> np.ndarray:
    if value > 49.9:
        print("Value cannot be bigger than 49.9")
        return image
    image_min = np.percentile(image, value)
    image_max = np.percentile(image, 100-value)
    image_changed = ((np.clip(image, image_min, image_max) - image_min) / (image_max - image_min)) * 255

    return image_changed.astype(np.uint8)

if __name__ == "__main__":
    path = 'C:\\Users\\zazon\\PycharmProjects\\ImageConverter\\ServerApp\\images_upload\\Widowmaker-Overwatch-Blizzard-8731049.jpg'
    path2 = 'C:\\Users\\zazon\\PycharmProjects\\ImageConverter\\ServerApp\\images_upload\\tiger-animals-cat-predator-preview.jpg'
    path3 = 'C:\\Users\\zazon\\PycharmProjects\\ImageConverter\\ServerApp\\images_upload\\GO48R.png'
    image_np = open_image(path)
    image_np_2 = open_image(path2)
    print(image_np.shape, image_np_2.shape)
    Image.fromarray(change_contrast(image_np_2)).show()
    #image_np_3 = open_image(path3)
    #Image.fromarray(image_np).show()
    #Image.fromarray(change_contrast(image_np)).show()
    #Image.fromarray(solarize_image(image_np, 70)).show()
    #Image.fromarray(solarize_image(image_np, 130)).show()
    #Image.fromarray(solarize_image(image_np, 160)).show()
    #Image.fromarray(solarize_image(image_np, 190)).show()
    #Image.fromarray(solarize_image(image_np, 240)).show()
    #Image.fromarray(solarize_image(image_np, 100)).show()
    #Image.fromarray(solarize_image(image_np, 130)).show()
    #Image.fromarray(solarize_image(image_np_2, 70)).show()
    #Image.fromarray(solarize_image(image_np_2, 100)).show()
    #Image.fromarray(solarize_image(image_np_2, 130)).show()
    #Image.fromarray(solarize_image(image_np_2, 160)).show()
    #Image.fromarray(solarize_image(image_np_2, 190)).show()
    #Image.fromarray(solarize_image(image_np_2, 220)).show()
    #Image.fromarray(solarize_image(image_np_2, 250)).show()
    #Image.fromarray(change_brightness(image_np, -30)).show()
    #Image.fromarray(change_brightness(image_np, 0)).show()
    #Image.fromarray(change_brightness(image_np, 30)).show()
    #Image.fromarray(change_brightness(image_np_2, 30)).show()