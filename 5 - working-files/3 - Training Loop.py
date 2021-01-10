import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import *

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
    def __init__(self, augmentation_dict, use_augmentations, train_dir, test_dir,
                 validation_split, target_size, batch_size, color_mode, shuffle, seed,
                 model, optimizer, enable_mixed_precision, loss, metrics,
                 tensorboard_logs_dir, model_checkpoint_filepath):

        self.augmentation_dict = augmentation_dict  # Datagenerator Param
        self.use_augmentations = use_augmentations  # Datagenerator Param - Boolean
        self.train_dir = train_dir  # Datagenerator Param - Train Images Dir
        self.test_dir = test_dir  # Datagenerator Param - Test Images Dir
        self.validation_split = validation_split  # Datagenerator Param
        self.target_size = target_size  # Datagenerator Param
        self.batch_size = batch_size  # Datagenerator Param
        self.color_mode = color_mode  # Datagenerator Param
        self.shuffle = shuffle  # Datagenerator Param
        self.seed = seed  # Datagenerator Param

        self.model = model  # Tensorflow Model
        self.optimizer = optimizer  # Keras Optimizer
        self.enable_mixed_precision = enable_mixed_precision   # Boolean - set true if NVIDIA gpu contains RT cores
        self.loss = loss    # Keras/custom loss function
        self.metrics = metrics
        self.tensorboard_logs_dir = tensorboard_logs_dir    # Directory to save tensorboard logs to
        self.model_checkpoint_filepath = model_checkpoint_filepath  # ########################################

    def compile_model(self, model, enable_mixed_precision):
        if enable_mixed_precision:
            optimizer = get_mixed_precision_opt(self.optimizer)
        else:
            optimizer = self.optimizer

        model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)

        return model

    def train_model(self, model, epochs, use_callbacks):
        callbacks = get_model_callbacks(self.tensorboard_logs_dir, self.model_checkpoint_filepath)

        start = time.time()

        if use_callbacks:
            history = model.fit(
                train_generator,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                callbacks=callbacks,
                validation_data=test_generator,
                validation_steps=validation_steps
            )

        else:
            history = model.fit(
                train_generator,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=test_generator,
                validation_steps=validation_steps
            )

        stop = time.time()
        total_training_time = stop - start

        return history, total_training_time


# ================================= TRAINING ANALYSIS ===================================

def export_model_stats(model_history, total_training_time, plot_path):
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
    fig.write_image(plot_path)


def save_model(trained_model, models_dir, model_name):
    trained_model = trained_model
    trained_model.save(f"{models_dir}/{model_name}.hdf5")


def load_saved_model(models_dir, model_name):
    model = load_model(f"{models_dir}/{model_name}.hdf5")

    return model


# =================================== MODEL EVALUATION ===================================
def evaluate_model(data_generator, models_dir, saved_model_name):
    saved_model = load_saved_model(models_dir, saved_model_name)

    scores = saved_model.evaluate(data_generator, steps=10)

    return scores


def print_scores(evaluated_scores):
    loss, accuracy = evaluated_scores

    print(f"""
    Evaluation Loss: {loss}
    Evaluation Accuracy: {accuracy}
    """)



