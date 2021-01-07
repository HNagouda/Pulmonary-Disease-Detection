from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.preprocessing.image import ImageDataGenerator

super_dataset_dir = "C:/Users/harsh/Desktop/Python/0 - Projects/Pulmonary-Disease-Detection/0 - Datasets/Super Dataset"

batchSize = 32

train_datagen = ImageDataGenerator(
        width_shift_range=0.25,
        height_shift_range=0.25,
        brightness_range=(0.25, 0.90),
        fill_mode='nearest',
        zoom_range=0.5,
        horizontal_flip=False,
        vertical_flip=False
)

train_generator = train_datagen.flow_from_directory(
        f"{super_dataset_dir}/train"
        batch_size = batchSize,
        class_mode = "categorical"
)