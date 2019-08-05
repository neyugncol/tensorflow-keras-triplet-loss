from base.base_data_loader import BaseDataLoader
import os
import math
import json
import numpy as np
from utils.image_processing import read_image, resize_image_keep_ratio, pad_image, get_augmenter


class DataLoader(BaseDataLoader):
    def __init__(self, config):
        super(DataLoader, self).__init__(config)
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
        self.test_category_ids = [cat['id'] for cat in self.categories if cat['split'] == 'val']

        if config.augment_images:
            self.augmenter = get_augmenter()

    def get_train_data(self):
        pass

    def get_test_data(self):
        pass

    def get_train_steps(self):
        num_of_example = len([ann for ann in self.annotations if ann['category_id'] in self.train_category_ids])
        num_of_steps = math.ceil(num_of_example / self.config.batch_size)

        return num_of_steps

    def get_val_steps(self):
        num_of_example = len([ann for ann in self.annotations if ann['category_id'] in self.val_category_ids])
        num_of_steps = math.ceil(num_of_example / self.config.batch_size)

        return num_of_steps

    def process_image(self, image, disable_augment=False):
        if self.config.augment_images and not disable_augment:
            image = self.augmenter.augment_image(image)
        image = resize_image_keep_ratio(image, (self.config.image_size, self.config.image_size))
        image = pad_image(image, (self.config.image_size, self.config.image_size))
        image = image / 255.0

        return image

    def get_train_generator(self):
        category_ids = np.pad(self.train_category_ids, [0, math.ceil(len(self.train_category_ids) / self.category_per_batch) * self.category_per_batch - len(self.train_category_ids)], mode='wrap')

        while True:
            for i in range(0, len(category_ids), self.category_per_batch):
                annotations = []
                start, end = i, i + self.category_per_batch
                for cat_id in category_ids[start:end]:
                    annotations.extend(np.random.choice(self.cat2ann[cat_id], math.ceil(self.config.batch_size / self.category_per_batch), replace=False))

                images, labels = [], []
                for ann in annotations[:self.config.batch_size]:
                    image = read_image(os.path.join(self.config.image_dir, ann['image_file']))
                    image = self.process_image(image)
                    label = ann['category_id']
                    images.append(image)
                    labels.append(label)

                yield np.array(images), np.array(labels)

            np.random.shuffle(category_ids)

    def get_val_generator(self):
        category_ids = np.pad(self.val_category_ids, [0, math.ceil(len(self.val_category_ids) / self.category_per_batch) * self.category_per_batch - len(self.val_category_ids)], mode='wrap')

        while True:
            for i in range(0, len(category_ids), self.category_per_batch):
                annotations = []
                start, end = i, i + self.category_per_batch
                for cat_id in category_ids[start:end]:
                    annotations.extend(np.random.choice(self.cat2ann[cat_id], math.ceil(self.config.batch_size / self.category_per_batch), replace=False))

                images, labels = [], []
                for ann in annotations[:self.config.batch_size]:
                    image = read_image(os.path.join(self.config.image_dir, ann['image_file']))
                    image = self.process_image(image, disable_augment=True)
                    label = ann['category_id']
                    images.append(image)
                    labels.append(label)

                yield np.array(images), np.array(labels)

    def get_test_generator(self):
        annotations = [ann for ann in self.annotations if ann['category_id'] in self.test_category_ids]
        for i in range(0, len(annotations), self.config.batch_size):
            j = i + self.config.batch_size if i + self.config.batch_size <= len(annotations) else len(annotations)

            images, labels = [], []
            for ann in annotations[i:j]:
                image = read_image(os.path.join(self.config.image_dir, ann['image_file']))
                image = self.process_image(image, disable_augment=True)
                label = ann['category_id']
                images.append(image)
                labels.append(label)

            yield np.array(images), np.array(labels)

    def get_test_references(self):
        images, labels = [], []
        for category in self.categories:
            if category['split'] != 'val':
                continue
            image = read_image(os.path.join(self.config.image_dir, category['reference_image']))
            image = self.process_image(image, disable_augment=True)
            label = category['id']
            images.append(image)
            labels.append(label)

        return np.array(images), np.array(labels)


