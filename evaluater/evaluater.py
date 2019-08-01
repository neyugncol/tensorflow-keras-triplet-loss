from comet_ml import Experiment
import numpy as np
from sklearn import metrics
from tqdm import tqdm


class ModelEvaluater():
    def __init__(self, model, data_loader, config):
        super(ModelEvaluater, self).__init__(model, data_loader, config)
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

        ref_embeddings = self.model.embedding_model.predict(ref_images, batch_size=self.config.batch_size)

        y_true = []
        y_pred = []
        for images, labels in tqdm(self.data_loader.get_test_generator(), desc='evaluating...'):
            predictions = self.model.model.predict([images, ref_embeddings], batch_size=self.config.batch_size)
            y_true.append(labels)
            y_pred.append(predictions)

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        accuracy = metrics.accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred, average='micro')
        recall = metrics.recall_score(y_true, y_pred, average='micro')
        f1_score = metrics.f1_score(y_true, y_pred, average='micro')

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
