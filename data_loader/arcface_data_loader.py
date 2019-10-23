from base.base_data_loader import BaseDataLoader
import os
import math
import json
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
import imgaug as ia
import imgaug.augmenters as iaa
from utils.image_processing import read_image, resize_image_keep_ratio, pad_image


class ArcFaceDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(ArcFaceDataLoader, self).__init__(config)
        self.config = config

        data = json.load(open(config.annotations_file, 'r'))
        self.annotations = data['annotations']
        self.categories = data['categories']
        self.cat2ann = {cat['id']: [] for cat in self.categories}
        for ann in self.annotations:
            cat_id = ann['category_id']
            self.cat2ann[cat_id].append(ann)

        self.category_per_batch = math.ceil(config.batch_size / config.example_per_category)

        self.train_category_ids = [cat['id'] for cat in self.categories if cat['split'] == 'train']
        self.val_category_ids = [cat['id'] for cat in self.categories if cat['split'] == 'val']

        self.class_mapping = {cat_id: i for i, cat_id in enumerate(self.train_category_ids)}

        if config.augment_images:
            self.augmenter = self.build_augmenter()

    def get_train_data(self):
        pass

    def get_test_data(self):
        pass

    def get_train_steps(self):
        if isinstance(self.config.steps_per_epoch, int):
            return self.config.steps_per_epoch

        num_of_example = round(len([ann for ann in self.annotations if ann['category_id'] in self.train_category_ids]) * 0.9)
        num_of_steps = math.ceil(num_of_example / self.config.batch_size)

        return num_of_steps

    def get_val_steps(self):
        num_of_example = round(len([ann for ann in self.annotations if ann['category_id'] in self.train_category_ids]) * 0.1)
        num_of_steps = math.ceil(num_of_example / self.config.batch_size)

        return num_of_steps

    def get_test_steps(self):
        num_of_example = round(len([ann for ann in self.annotations if ann['category_id'] in self.val_category_ids]) * 0.9)
        num_of_steps = math.ceil(num_of_example / self.config.batch_size)

        return num_of_steps

    def process_image(self, image):
        image = resize_image_keep_ratio(image, (self.config.image_size, self.config.image_size))
        image = pad_image(image, (self.config.image_size, self.config.image_size))
        image = image / 255.0

        return image

    def augment_image(self, image):
        if self.config.augment_images:
            image = self.augmenter.augment_image(image)

        return image

    def get_train_generator(self):
        annotations = np.concatenate([anns[:round(len(anns) * 0.9)] for cat_id, anns in self.cat2ann.items() if cat_id in self.train_category_ids])

        while True:
            np.random.shuffle(annotations)

            for i in range(len(annotations) // self.config.batch_size):
                images = []
                labels = []
                for j in range(self.config.batch_size):
                    annotation = annotations[i * self.config.batch_size + j]
                    image = read_image(os.path.join(self.config.image_dir, annotation['image_file']))
                    image = self.augment_image(image)
                    image = self.process_image(image)
                    label = self.class_mapping[annotation['category_id']]
                    images.append(image)
                    labels.append(label)

                images = np.array(images)
                labels = to_categorical(np.array(labels), num_classes=len(self.train_category_ids))

                yield [images, labels], labels

    def get_val_generator(self):
        annotations = np.concatenate([anns[round(len(anns) * 0.9):] for cat_id, anns in self.cat2ann.items() if cat_id in self.train_category_ids])

        while True:

            for i in range(len(annotations) // self.config.batch_size):
                images = []
                labels = []
                for j in range(self.config.batch_size):
                    annotation = annotations[i * self.config.batch_size + j]
                    image = read_image(os.path.join(self.config.image_dir, annotation['image_file']))
                    image = self.process_image(image)
                    label = self.class_mapping[annotation['category_id']]
                    images.append(image)
                    labels.append(label)

                images = np.array(images)
                labels = to_categorical(np.array(labels), num_classes=len(self.train_category_ids))

                yield [images, labels], labels

    def get_test_generator(self):
        annotations = np.concatenate([anns[10:] for cat_id, anns in self.cat2ann.items() if cat_id in self.val_category_ids])

        while True:
            for i in range(len(annotations) // self.config.batch_size):
                images = []
                labels = []
                for j in range(self.config.batch_size):
                    annotation = annotations[i * self.config.batch_size + j]
                    image = read_image(os.path.join(self.config.image_dir, annotation['image_file']))
                    image = self.process_image(image)
                    label = annotation['category_id']
                    images.append(image)
                    labels.append(label)

                images = np.array(images)
                labels = np.array(labels)

                yield images, labels

    def get_reference_data(self):
        annotations = np.concatenate([anns[:10] for cat_id, anns in self.cat2ann.items() if cat_id in self.val_category_ids])

        images, labels = [], []
        for ann in annotations:
            image = read_image(os.path.join(self.config.image_dir, ann['image_file']))
            image = self.process_image(image)
            label = ann['category_id']
            images.append(image)
            labels.append(label)

        return np.array(images), np.array(labels)

    def build_augmenter(self):
        augmenter = iaa.Sometimes(0.5, iaa.Sequential([
            iaa.OneOf([
                iaa.OneOf([
                    iaa.Crop(percent=((0.1, 0.3), 0, 0, 0), keep_size=False),
                    iaa.Crop(percent=(0, (0.1, 0.3), 0, 0), keep_size=False),
                    iaa.Crop(percent=(0, 0, (0.1, 0.3), 0), keep_size=False),
                    iaa.Crop(percent=(0, 0, 0, (0.1, 0.3)), keep_size=False),
                    iaa.Crop(percent=((0.05, 0.1), 0, (0.05, 0.1), 0), keep_size=False),
                    iaa.Crop(percent=(0, (0.05, 0.1), 0, (0.05, 0.1)), keep_size=False)
                ]),
                iaa.Affine(
                    scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    order=[0, 1]
                ),
                iaa.Noop()
            ]),
            iaa.Sometimes(0.3, iaa.Rot90(
                k=[1, 2, 3],
                keep_size=False
            ))
        ]))

        return augmenter


