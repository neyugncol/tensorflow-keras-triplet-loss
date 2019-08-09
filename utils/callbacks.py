from tensorflow.python.keras import callbacks
import numpy as np
from sklearn import metrics
from scipy.spatial.distance import cdist
from tqdm import tqdm
from utils.image_processing import get_confusion_matrix_figure


class Evaluater(callbacks.Callback):

    def __init__(self, eval_data, eval_steps, ref_data, config, comet_experiment=None):
        super(Evaluater, self).__init__()
        self.eval_data = eval_data
        self.eval_steps = eval_steps
        self.ref_data = ref_data
        self.config = config
        self.comet_experiment = comet_experiment
        self.train_step = 0

    def on_train_batch_end(self, batch, logs=None):
        self.train_step = self.train_step + 1

    def on_epoch_end(self, epoch, logs=None):
        ref_images, ref_labels = self.ref_data
        ref_embeddings = self.model.predict(ref_images, batch_size=self.config.batch_size)
        eval_embeddings = []
        eval_labels = []
        for step, (images, labels) in tqdm(enumerate(self.eval_data), desc='Evaluate', total=self.eval_steps - 1, ncols=70):
            embeddings = self.model.predict(images)
            eval_embeddings.append(embeddings)
            eval_labels.append(labels)
            if step >= self.eval_steps - 1:
                break

        eval_embeddings = np.concatenate(eval_embeddings)
        eval_labels = np.concatenate(eval_labels)
        eval_categories = np.unique(eval_labels)

        pairwise_distance = cdist(eval_embeddings, ref_embeddings, metric=self.config.distance_metric)
        predictions = ref_labels[np.argmin(pairwise_distance, axis=1)]

        result = {
            'val_accuracy': metrics.accuracy_score(eval_labels, predictions),
            'val_precision': metrics.precision_score(eval_labels, predictions, labels=eval_categories, average='macro'),
            'val_recall': metrics.recall_score(eval_labels, predictions, labels=eval_categories, average='macro'),
            'val_f1_score': metrics.f1_score(eval_labels, predictions, labels=eval_categories, average='macro')
        }

        print(' Result: {}'.format(' - '.join(['{}: {}'.format(key, value) for key, value in result.items()])))

        if logs is not None:
            logs.update(result)
        if self.comet_experiment is not None:
            self.comet_experiment.log_metrics(result, step=self.train_step)

            cf_mat = metrics.confusion_matrix(eval_labels, predictions, labels=eval_categories)
            cf_figure = get_confusion_matrix_figure(cf_mat, eval_categories)
            self.comet_experiment.log_figure('confusion_matrix', cf_figure)
