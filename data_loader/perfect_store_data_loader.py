from .triplet_loss_data_loader import TripletLossDataLoader
import os
import math
import json
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from utils.image_processing import read_image, resize_image_keep_ratio, pad_image


class PerfectStoreDataLoader(TripletLossDataLoader):
    def __init__(self, config):
        super(PerfectStoreDataLoader, self).__init__(config)

        self.cat2reference = {cat['id']: {'category_id': cat['id'],
                                          'image_file': cat['reference_image'],
                                          'is_reference': True}
                              for cat in self.categories}
        # self.cat2rel = {}
        # for cat in self.categories:
        #     cat_id = cat['id']
        #     if cat_id not in self.cat2rel:
        #         self.cat2rel[cat_id] = []

        if self.config.use_hierarchical_triplet_loss:
            self.brands = list(set(cat['brand'] for cat in self.categories))
            self.packagings = list(set(cat['packaging'] for cat in self.categories))
            self.groups = list(set(cat['product_group'] for cat in self.categories))
            for cat_id, annotations in self.cat2ann.items():
                category = next(cat for cat in self.categories if cat['id'] == cat_id)
                brand_id = self.brands.index(category['brand'])
                packaging_id = self.packagings.index(category['packaging'])
                group_id = self.groups.index(category['product_group'])
                ref_ann = self.cat2reference[cat_id]
                ref_ann['brand_id'] = brand_id
                ref_ann['packaging_id'] = packaging_id
                ref_ann['group_id'] = group_id
                for ann in annotations:
                    ann['brand_id'] = brand_id
                    ann['packaging_id'] = packaging_id
                    ann['group_id'] = group_id
        self.reference_augmenter = self.build_reference_augmenter()

    def augment_image(self, image, is_reference):
        if is_reference:
            image = self.reference_augmenter.augment_image(image)
        elif self.config.augment_images:
            image = self.augmenter.augment_image(image)
        return image

    def process_reference_image(self, image):
        image = resize_image_keep_ratio(image, (self.config.image_size // 2, self.config.image_size // 2))
        image = resize_image_keep_ratio(image, (self.config.image_size, self.config.image_size))
        image = pad_image(image, (self.config.image_size, self.config.image_size))
        image = image / 255.0

        return image

    def get_train_generator(self):
        category_ids = np.pad(self.train_category_ids, [0, math.ceil(
            len(self.train_category_ids) / self.category_per_batch) * self.category_per_batch - len(
            self.train_category_ids)], mode='wrap')

        while True:
            for i in range(0, len(category_ids), self.category_per_batch):
                annotations = []
                start, end = i, i + self.category_per_batch
                for cat_id in category_ids[start:end]:
                    annotations.append(self.cat2reference[cat_id])
                    num_samples = math.ceil(self.config.batch_size / self.category_per_batch) - 1
                    num_samples = num_samples if num_samples <= len(self.cat2ann[cat_id]) else len(self.cat2ann[cat_id])
                    annotations.extend(np.random.choice(self.cat2ann[cat_id],
                                                        num_samples,
                                                        replace=False))

                images, labels = [], []
                for ann in annotations[:self.config.batch_size]:
                    image = read_image(os.path.join(self.config.image_dir, ann['image_file']))
                    image = self.augment_image(image, ann.get('is_reference', False))
                    image = self.process_image(image)
                    if self.config.use_hierarchical_triplet_loss:
                        label = [ann['category_id'], ann['brand_id'], ann['packaging_id'], ann['group_id']]
                    else:
                        label = ann['category_id']
                    images.append(image)
                    labels.append(label)

                yield np.array(images), np.array(labels)

            np.random.shuffle(category_ids)

    def get_reference_data(self):
        images, labels = [], []
        for category in self.categories:
            image = read_image(os.path.join(self.config.image_dir, category['reference_image']))
            image = self.process_reference_image(image)
            label = category['id']
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

    def build_reference_augmenter(self):
        augmenter = iaa.Sometimes(0.7, iaa.Sequential([
            iaa.Sequential([
                iaa.Resize({'longer-side': (144, 48), 'shorter-side': 'keep-aspect-ratio'}, interpolation=ia.ALL),
                iaa.Resize({'longer-side': self.config.image_size, 'shorter-side': 'keep-aspect-ratio'},
                           interpolation=ia.ALL)
            ]),
            iaa.Affine(
                scale=(0.7, 1.1),
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-15, 15),
                order=[0, 1],
                mode=ia.ALL,
            ),
            iaa.Sometimes(0.3, iaa.Rot90([1, 2, 3], keep_size=False)),

            iaa.OneOf([
                iaa.Multiply((0.2, 1.2)),
                iaa.GammaContrast((1.0, 1.7)),
            ]),

            iaa.OneOf([
                iaa.Noop(),
                iaa.JpegCompression(compression=(85, 95)),
                iaa.GaussianBlur(sigma=(0.75, 2.25)),
                iaa.MotionBlur(k=(10, 15))
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
            ])
        ]))

        return augmenter

