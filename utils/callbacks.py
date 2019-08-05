import tensorflow as tf
from tensorflow.python.keras import callbacks
import numpy as np
from sklearn import metrics
from scipy.spatial.distance import cdist


class Evaluater(callbacks.Callback):

    def __init__(self, ref_data, config, comet_experiment=None):
        super(Evaluater).__init__(self)
        self.ref_data = ref_data
        self.config = config
        self.comet_experiment = comet_experiment

    def on_epoch_end(self, epoch, logs=None):
        ref_images, ref_labels = self.ref_data
        ref_embeddings = self.model.predict(ref_images, batch_size=self.config.bacth_size)
        eval_embeddings = []
        eval_labels = []
        for images, labels in self.validation_data:
            embeddings = self.model.predict(images)
            eval_embeddings.append(embeddings)
            eval_labels.append(labels)

        eval_embeddings = np.concatenate(eval_embeddings)
        eval_labels = np.concatenate(eval_labels)

        pairwise_distance = cdist(eval_embeddings, ref_embeddings, metric=self.config.distance_metric)
        predictions = ref_labels[np.argmin(pairwise_distance, axis=1)]

        results = {
            'val_accuracy': metrics.accuracy_score(eval_labels, predictions),
            'val_precision': metrics.precision_score(eval_labels, predictions, average='macro'),
            'val_recall': metrics.recall_score(eval_labels, predictions, average='macro'),
            'val_f1_score': metrics.f1_score(eval_labels, predictions, average='macro')
        }

        if logs is not None:
            logs.update(results)
        if self.comet_experiment is not None:
            step = epoch * self.config.bacth_size
            self.comet_experiment.log_metrics(results, step=step)
