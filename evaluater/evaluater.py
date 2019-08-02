from comet_ml import Experiment
import numpy as np
from scipy.spatial.distance import cdist
from sklearn import metrics
from tqdm import tqdm


class ModelEvaluater():
    def __init__(self, model, data_loader, config):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.init_logger()

    def init_logger(self):
        if hasattr(self.config, "comet_api_key"):
            experiment = Experiment(api_key=self.config.comet_api_key, project_name=self.config.experiment_name)
            experiment.log_parameters(self.config)
            self.logger = experiment

    def evaluate(self):

        ref_images, ref_labels = self.data_loader.get_test_references()

        ref_embeddings = self.model.predict(ref_images, batch_size=self.config.batch_size)

        eval_embeddings = []
        eval_labels = []
        for images, labels in tqdm(self.data_loader.get_test_generator(), desc='evaluating...'):
            embeddings = self.model.predict(images, batch_size=self.config.batch_size)
            eval_embeddings.append(embeddings)
            eval_labels.append(labels)

        eval_embeddings = np.concatenate(eval_embeddings)
        eval_labels = np.concatenate(eval_labels)

        pairwise_distance = cdist(eval_embeddings, ref_embeddings)
        predictions = ref_labels[np.argmin(pairwise_distance, axis=1)]

        accuracy = metrics.accuracy_score(eval_labels, predictions)
        precision = metrics.precision_score(eval_labels, predictions, average='micro')
        recall = metrics.recall_score(eval_labels, predictions, average='micro')
        f1_score = metrics.f1_score(eval_labels, predictions, average='micro')

        result = {'accuracy': accuracy,
                  'precision': precision,
                  'recall': recall,
                  'f1_score': f1_score}

        print('Evaluation complete: {}'.format(result))

        self.logger(result)

        self.accuracy.extend(accuracy)
        self.precision.extend(precision)
        self.recall.extend(recall)
        self.f1_score.extend(f1_score)
