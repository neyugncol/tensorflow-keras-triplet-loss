from tensorflow.python.keras import callbacks
from tensorflow.python.keras import backend as K
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
            'val_balanced_accuracy': metrics.balanced_accuracy_score(eval_labels, predictions),
            'val_precision': metrics.precision_score(eval_labels, predictions, labels=eval_categories, average='macro'),
            'val_recall': metrics.recall_score(eval_labels, predictions, labels=eval_categories, average='macro'),
            'val_f1_score': metrics.f1_score(eval_labels, predictions, labels=eval_categories, average='macro'),
            'val_cohen_kappa': metrics.cohen_kappa_score(eval_labels, predictions, labels=eval_categories)
        }

        print(' Result: {}'.format(' - '.join(['{}: {}'.format(key, value) for key, value in result.items()])))

        if logs is not None:
            logs.update(result)
        if self.comet_experiment is not None:
            self.comet_experiment.log_metrics(result, step=self.train_step)

            cf_mat = metrics.confusion_matrix(eval_labels, predictions, labels=ref_labels)
            cf_mat = cf_mat.astype(np.float)
            cf_total = cf_mat.sum(axis=0)[:, np.newaxis]
            cf_mat = np.divide(cf_mat, cf_total, out=np.zeros_like(cf_mat), where=cf_total!=0)
            eval_category_ids = np.searchsorted(ref_labels, eval_categories)
            cf_mat = cf_mat[eval_category_ids][:, eval_category_ids]
            cf_figure = get_confusion_matrix_figure(cf_mat, eval_categories)
            self.comet_experiment.log_figure('confusion_matrix', cf_figure)


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.
    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.
    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))
