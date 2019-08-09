import cv2
import numpy as np
import imgaug
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

