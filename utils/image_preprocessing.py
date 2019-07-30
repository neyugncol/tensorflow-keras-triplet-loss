import cv2
import numpy as np
import imgaug.augmenters as iaa


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


def get_augmenter():
    augmenter = iaa.Sometimes(0.7, iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),

        iaa.OneOf([
            iaa.Noop(),
            iaa.Multiply((0.2, 2.0)),
            iaa.GammaContrast((0.5, 1.7)),
        ]),

        iaa.OneOf([
            iaa.Noop(),
            iaa.JpegCompression(compression=(85, 95)),
            iaa.MotionBlur(k=(10, 15)),
            iaa.CoarsePepper(p=0.2, size_percent=0.1)
        ]),

        iaa.OneOf([
            iaa.Noop(),
            iaa.OneOf([
                iaa.Crop(percent=((0.2, 0.5), 0, 0, 0), keep_size=False),
                iaa.Crop(percent=(0, (0.2, 0.5), 0, 0), keep_size=False),
                iaa.Crop(percent=(0, 0, (0.2, 0.5), 0), keep_size=False),
                iaa.Crop(percent=(0, 0, 0, (0.2, 0.5)), keep_size=False),
                iaa.Crop(percent=((0.1, 0.3), 0, (0.1, 0.3), 0), keep_size=False),
                iaa.Crop(percent=(0, (0.1, 0.3), 0, (0.1, 0.3)), keep_size=False)
            ]),
            iaa.Crop(percent=(0.1, 0.2)),
            iaa.PerspectiveTransform(0.1)
        ]),
    ]))
    return augmenter

