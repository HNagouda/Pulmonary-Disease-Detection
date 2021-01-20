from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import *
from tensorflow.python.keras.layers.advanced_activations import *
from tensorflow.keras.initializers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.models import Model


def pulmonet1(input_shape):
    model = Sequential()

    model.add(Conv2D(32, (5, 5), input_shape=input_shape, padding='same'))
    model.add(PReLU(alpha_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same'))
    model.add(PReLU(alpha_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(128, (3, 3), input_shape=input_shape, padding='same'))
    model.add(PReLU(alpha_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(PReLU(alpha_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(256, (3, 3), input_shape=input_shape, padding='same'))
    model.add(PReLU(alpha_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(256, (3, 3), input_shape=input_shape, padding='same'))
    model.add(PReLU(alpha_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Flatten())

    model.add(Dense(14, activation='softmax'))

    return model
