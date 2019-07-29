from comet_ml import Experiment
from base.base_trainer import BaseTrain
import os
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard


class TripletLossModelTrainer(BaseTrain):
    def __init__(self, model, data_loader, config):
        super(TripletLossModelTrainer, self).__init__(model, data_loader, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        if hasattr(self.config, "comet_api_key"):
            experiment = Experiment(api_key=self.config.comet_api_key, project_name=self.config.experiment_name)
            experiment.log_parameters(self.config)
            self.callbacks.append(experiment.get_keras_callback())

        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.experiment_name),
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
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['accuracy'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_accuracy'])
