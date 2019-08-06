import cv2
import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


PADDING_VALUE = [0, 0, 0]


def read_image(image_path):
    image = cv2.imread(image_path)
    assert image is not None, 'image file not found'
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def resize_image_keep_ratio(image, max_size):
    height, width, _ = np.shape(image)
    max_width, max_height = max_size

    width_scale = max_width / width
    height_scale = max_height / height
    scale = min(width_scale, height_scale)

    new_size = round(width * scale), round(height * scale)

    image = cv2.resize(image, new_size)

    return image


def pad_image(image, new_size, padding_value=PADDING_VALUE):
    height, width, _ = np.shape(image)
    new_width, new_height = new_size
    assert width <= new_width and height <= new_height, 'new size must be equal or greater than image size'

    delta_w = new_width - width
    delta_h = new_height - height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_value)

    return image


def get_confusion_matrix_figure(confusion_matrix, labels):
    confusion_matrix = confusion_matrix.astype(np.float) / confusion_matrix.sum(axis=1)[:, np.newaxis]
    df = pd.DataFrame(confusion_matrix, labels, labels)
    ax = sns.heatmap(df, cmap=plt.cm.Blues, cbar=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set(xlabel='Predicted label', ylabel='True label', title='Confusion matrix')

    return ax.get_figure()


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
            iaa.PerspectiveTransform(0.1)
        ]),
    ]))
    return augmenter

