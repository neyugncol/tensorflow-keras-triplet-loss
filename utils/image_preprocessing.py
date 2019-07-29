import cv2
import numpy as np


PADDING_VALUE = [0, 0, 0]


def load_image(image_path):
    image = cv2.imread(image_path)
    assert image is not None, 'Image file not found.'
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def resize_image_with_padding(image, new_size):
    width, height, _ = np.shape(image)
    max_size = max(width, height)
    ratio = new_size / max_size
    width, height = int(width * ratio), int(height * ratio)
    image = cv2.resize(image, (width, height))

    delta_w = new_size - width
    delta_h = new_size - height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=PADDING_VALUE)

    return image

