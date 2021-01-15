import os
import Train_Loops
import Model_Architectures
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

base_path = "C:/Users/harsh/Desktop/Python/0 - Projects/Pulmonary-Disease-Detection"

# -----------------------------------------------------------------------------------------------------------
# ================================= PATHS, PARAMETERS, AND MODEL TUNING =====================================
# -----------------------------------------------------------------------------------------------------------

basic_configs = {
    # RUNTIME NAME
    'runtime_name': "check",  # MUST change this every time the code is run

    # PATHS
    'image_data_dir': os.path.join(base_path, '0_Datasets', 'Super Dataset', '256x256'),
    'model_checkpoint_dir': os.path.join(base_path, '2_Model Checkpoints'),
    'plot_model_stats_dir': os.path.join(base_path, '3_Model Progress Plots'),
    'save_models_dir': os.path.join(base_path, '4_Saved_Models'),
    'train_models_dir': os.path.join(base_path, 'Model_Architectures')
}

model_configs = {
    'input_shape': (256, 256, 3),

    # Adam Optimizer arguments
    'adam_opt': {
        'lr': 0.001,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-07,
        'amsgrad': False
    }
}

model_to_train = Model_Architectures.LoopTester.loop_tester(model_configs['input_shape'])

train_loop_configs = {
    'model_name': 'LoopTester',
    'model': model_to_train,
    'optimizer': Adam(**model_configs['adam_opt']),
    'enable_mixed_precision': True,
    'loss': 'categorical_crossentropy',
    'metrics': ['categorical_accuracy'],
    'use_callbacks': True,
    'epochs': 1,
    'steps_per_epoch': 10
}

augmentation_dict = {
    'width_shift_range': 0.25,
    'height_shift_range': 0.25,
    'brightness_range': (0.25, 0.90),
    'fill_mode': 'nearest',
    'zoom_range': 0.5,
    'horizontal_flip': False,
    'vertical_flip': False
}

datagen_configs = {
    'augmentation_dict': augmentation_dict,
    'use_augmentations': True,
    'train_dir': os.path.join(basic_configs['image_data_dir'], 'train'),
    'test_dir': os.path.join(basic_configs['image_data_dir'], 'test'),
    'validation_split': 0.3,
    'target_size': (model_configs['input_shape'][0], model_configs['input_shape'][1]),
    'batch_size': 1,
    'color_mode': 'grayscale',
    'shuffle': True,
    'seed': 777
}

configs_dict = {
    'basic_configs': basic_configs,
    'model_configs': model_configs,
    'train_loop_configs': train_loop_configs,
    'augmentation_dict': augmentation_dict,
    'datagen_configs': datagen_configs
}

# -----------------------------------------------------------------------------------------------------------
# =================== LOGGER FUNCTION TO CREATE A LOG OF THE MODEL, CONFIGS, AND ACCURACY ===================
# -----------------------------------------------------------------------------------------------------------
logs_dir = os.path.join(base_path, '7_RuntimeLogs_and_Configs')


def logger(logs_dir, runtime_name, configs_dict, model_scores):
    log_path = f"{os.path.join(logs_dir, runtime_name)}.txt"

    with open(log_path, 'w') as log:
        log.writelines(['-' * 80, '\n'])
        log.write(f"Runtime Log for runtime: {runtime_name} \n")
        log.writelines(['-' * 80, '\n\n'])

        for name, dict in configs_dict.items():
            log.writelines([f'{name}: ', '\n'])

            for item_name, item in dict.items():
                if item_name == 'model':
                    pass
                else:
                    log.writelines([f'      {item_name}: ', f'{item}', '\n'])

            log.write('\n')

        log.writelines(['\n', "Model Stats: ", '\n'])
        log.writelines([f"    Model Loss: {model_scores[0]}", '\n'])
        log.writelines([f"    Model Accuracy: {model_scores[1]}"])

        log.close()


# -----------------------------------------------------------------------------------------------------------
# ============================================ PROJECT EXECUTION ============================================
# -----------------------------------------------------------------------------------------------------------
train_loop = Train_Loops.generic_train_loop.GenericTrainLoop(
    datagen_params=datagen_configs, plot_dir=basic_configs['plot_model_stats_dir'],
    runtime_name=basic_configs['runtime_name'], saved_models_dir=basic_configs['save_models_dir'],
    model_checkpoint_dir=basic_configs['model_checkpoint_dir'], **train_loop_configs
)

model_scores = train_loop.run_loop()

logger(logs_dir, basic_configs['runtime_name'], configs_dict, model_scores)
