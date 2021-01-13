import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import load_model

from Datagen import ImgDataGenerator

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)


# -----------------------------------------------------------------------------------------------------------
# =========================================== MODEL CALLBACKS ===============================================
# -----------------------------------------------------------------------------------------------------------

def reducelronplateau():
    reduce_lr_on_plateau = ReduceLROnPlateau(
        monitor='loss', factor=0.05,
        patience=5, verbose=1, mode='auto',
        min_delta=0.0001, cooldown=0, min_lr=0.001)

    return reduce_lr_on_plateau


def tensorboard(logs_dir):
    tensor_board = TensorBoard(
        log_dir=logs_dir, histogram_freq=0,
        write_graph=True, write_images=False,
        update_freq='epoch', profile_batch=2,
        embeddings_freq=0, embeddings_metadata=None)

    return tensor_board


def modelcheckpoint(checkpoint_filepath):
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True, save_best_only=True,
        monitor='val_acc', mode='max')

    return model_checkpoint


def earlystopping():
    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=0,
        verbose=0, mode='auto', baseline=None,
        restore_best_weights=False)

    return early_stopping


def get_model_callbacks(tensorboard_logs_dir, model_checkpoint_filepath):
    callbacks = [
        reducelronplateau(),
        tensorboard(tensorboard_logs_dir),
        modelcheckpoint(model_checkpoint_filepath),
        earlystopping()
    ]

    return callbacks


# -----------------------------------------------------------------------------------------------------------
# ============================================= MODEL OPTIMIZERS ============================================
# -----------------------------------------------------------------------------------------------------------


def get_mixed_precision_opt(optimizer):
    return tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)


# =================================== MODEL COMPILING ===================================

class ModelTrainLoop:
    def __init__(self, datagen_params, model, optimizer, enable_mixed_precision, loss, metrics,
                 tensorboard_logs_dir, model_checkpoint_filepath, use_callbacks, epochs, steps_per_epoch,
                 validation_steps, plot_dir, runtime_name, saved_models_dir, model_name):
        self.datagen_params = datagen_params
        self.model = model  # Tensorflow Model
        self.optimizer = optimizer  # Keras Optimizer
        self.enable_mixed_precision = enable_mixed_precision   # Boolean - set true if NVIDIA gpu contains RT cores
        self.loss = loss    # Keras/custom loss function
        self.metrics = metrics
        self.tensorboard_logs_dir = tensorboard_logs_dir    # Directory to save tensorboard logs to
        self.model_checkpoint_filepath = model_checkpoint_filepath  # ########################################
        self.use_callbacks = use_callbacks
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.plot_dir = plot_dir
        self.runtime_name = runtime_name
        self.saved_models_dir = saved_models_dir
        self.model_name = model_name

    def compile_model(self):
        if self.enable_mixed_precision:
            optimizer = get_mixed_precision_opt(self.optimizer)
        else:
            optimizer = self.optimizer

        self.model = self.model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)

        return self.model

    def import_generators(self):
        train_generator, test_generator = ImgDataGenerator(**self.datagen_params)

        return [train_generator, test_generator]

    def train_model(self, train_generator):
        callbacks = get_model_callbacks(self.tensorboard_logs_dir, self.model_checkpoint_filepath)

        start = time.time()

        if self.use_callbacks:
            history = self.model.fit(
                train_generator,
                epochs=self.epochs,
                steps_per_epoch=self.steps_per_epoch,
                callbacks=callbacks,
                validation_steps=self.validation_steps
            )

        else:
            history = self.model.fit(
                train_generator,
                epochs=self.epochs,
                steps_per_epoch=self.steps_per_epoch,
                validation_steps=self.validation_steps
            )

        stop = time.time()
        total_training_time = stop - start

        return history, total_training_time

# ================================= TRAINING ANALYSIS ===================================

    def export_model_stats(self, model_history, total_training_time):
        plot_title = f"""Visualizing Model Progress
        Total Training Time: {total_training_time} seconds"""

        history = model_history.history

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=['Loss', 'Accuracy'])

        fig.add_trace(go.Scatter(x=np.arange(1, 11), y=history['val_loss'],
                                 mode='lines+markers', name='Loss'),
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=np.arange(1, 11), y=history['val_binary_accuracy'],
                                 mode='lines+markers', name='Accuracy'),
                      row=1, col=2)

        fig.update_xaxes(title_text='Epochs', row=1, col=1)
        fig.update_xaxes(title_text='Epochs', row=1, col=2)

        fig.update_layout(title=plot_title)
        fig.write_image(os.path.join(self.plot_dir, self.runtime_name))

    def save_model(self):
        self.model.save(f"{self.saved_models_dir}/{self.model_name}.hdf5")

    def load_saved_model(self):
        model = load_model(f"{self.saved_models_dir}/{self.model_name}.hdf5")

        return model

# =================================== MODEL EVALUATION ===================================
    def evaluate_model(self, test_generator):
        saved_model = self.load_saved_model()
        scores = saved_model.evaluate(test_generator, steps=10)

        return scores

    def print_scores(self, evaluated_scores):
        loss, accuracy = evaluated_scores

        print(f"""
        Evaluation Loss: {loss}
        Evaluation Accuracy: {accuracy}
        """)

    def run_loop(self):
        self.compile_model()
        train_generator, test_generator = self.import_generators()

        history, total_training_time = self.train_model(train_generator)
        self.export_model_stats(history, total_training_time)

        self.save_model()

        scores = self.evaluate_model(test_generator)
        self.print_scores(scores)




