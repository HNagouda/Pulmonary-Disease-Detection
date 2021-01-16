from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import *
from tensorflow.python.keras.layers.advanced_activations import *
from tensorflow.keras.initializers import *
from tensorflow.keras.regularizers import *


def loop_tester(input_shape):
    model = Sequential()

    model.add(Conv2D(16, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform', bias_initializer='zeros',
                     kernel_regularizer='l2', padding='valid', data_format="channels_last",
                     input_shape=input_shape))
    # model.add(PReLU(alpha_initializer=VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    model.add(Activation(relu))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(GaussianNoise(0.1))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform', kernel_regularizer='l2',
                     padding='same'))
    # model.add(PReLU(alpha_initializer=VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    model.add(Activation(relu))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Dropout(0.5, noise_shape=None, seed=None))

    model.add(Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform', kernel_regularizer='l2',
                     padding='same'))
    # model.add(PReLU(alpha_initializer=VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    model.add(Activation(relu))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(GaussianNoise(0.1))

    model.add(Flatten())

    model.add(Dense(512, kernel_initializer='lecun_normal', kernel_regularizer='l2', activity_regularizer='l2'))
    model.add(Activation(relu))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(AlphaDropout(0.5, noise_shape=None, seed=None))

    model.add(Dense(14))
    model.add(Activation('softmax'))

    return model
