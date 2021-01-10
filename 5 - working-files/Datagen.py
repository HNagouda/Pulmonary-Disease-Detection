from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ImgDataGenerator:
    def __init__(self, augmentation_dict, use_augmentations, train_dir, test_dir,
                 validation_split, target_size, batch_size, color_mode, shuffle, seed):

        self.augmentation_dict = augmentation_dict
        self.use_augmentations = use_augmentations
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.validation_split = validation_split
        self.target_size = target_size
        self.batch_size = batch_size
        self.color_mode = color_mode
        self.shuffle = shuffle
        self.seed = seed

    def construct_datagens(self):
        if self.use_augmentations:
            train_datagen = ImageDataGenerator(self.validation_split, **self.augmentation_dict)
        else:
            train_datagen = ImageDataGenerator(self.validation_split)

        test_datagen = ImageDataGenerator()

        train_datagen = train_datagen.flow_from_directory(
            directory=self.train_dir,
            target_size=self.target_size,
            color_mode=self.color_mode,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            class_mode='binary'
        )

        test_datagen = test_datagen.flow_from_directory(
            directory=self.test_dir,
            target_size=self.target_size,
            color_mode=self.color_mode,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            class_mode='binary'
        )

        return [train_datagen, test_datagen]
