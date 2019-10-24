from comet_ml import Experiment
from base.base_trainer import BaseTrain
from utils.callbacks import ArcFaceEvaluater
import os
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau


class ArcFaceModelTrainer(BaseTrain):
    def __init__(self, model, predict_model, data_loader, config):
        super(ArcFaceModelTrainer, self).__init__(model, data_loader, config)
        self.predict_model = predict_model
        self.callbacks = []
        self.init_callbacks()

    def init_callbacks(self):
        experiment = None
        if hasattr(self.config, "comet_api_key"):
            experiment = Experiment(api_key=self.config.comet_api_key, project_name=self.config.experiment_name)
            experiment.log_parameters(self.config)
            self.callbacks.append(experiment.get_keras_callback())

        self.callbacks.append(
            ArcFaceEvaluater(
                model=self.predict_model,
                eval_data=self.data_loader.get_test_generator(),
                eval_steps=self.data_loader.get_test_steps(),
                ref_data=self.data_loader.get_reference_data(),
                config=self.config,
                comet_experiment=experiment
            )
        )

        self.callbacks.append(
            ReduceLROnPlateau(
                monitor=self.config.checkpoint_monitor,
                factor=self.config.lr_reduce_factor,
                patience=self.config.lr_reduce_patience,
                verbose=1,
                mode=self.config.checkpoint_mode
            )
        )

        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.checkpoint_dir, '%s-{epoch:02d}-{val_acc:.2f}.hdf5' % self.config.experiment_name),
                monitor=self.config.checkpoint_monitor,
                mode=self.config.checkpoint_mode,
                save_best_only=self.config.checkpoint_save_best_only,
                save_weights_only=self.config.checkpoint_save_weights_only,
                verbose=self.config.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.tensorboard_log_dir,
                write_graph=self.config.tensorboard_write_graph,
            )
        )

    def train(self):
        history = self.model.fit_generator(
            self.data_loader.get_train_generator(),
            steps_per_epoch=self.data_loader.get_train_steps(),
            epochs=self.config.num_epochs,
            verbose=self.config.verbose_training,
            validation_data=self.data_loader.get_val_generator(),
            validation_steps=self.data_loader.get_val_steps(),
            callbacks=self.callbacks,
        )

        return history
