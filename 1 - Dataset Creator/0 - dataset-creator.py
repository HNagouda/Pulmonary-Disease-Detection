import os
import csv
import cv2
import time
import random
import numpy as np
import pandas as pd
from glob import glob
from joblib import Parallel, delayed

# -----------------------------------------------------------------------------
#     ========================== DIRECTORIES ===========================

# directory of "NIH multiple disease dataset"
nih_dir = "C:/Users/harsh/Desktop/Python/0 - Projects/Pulmonary-Disease-Detection/0 - Datasets/NIH Multiple Disease Dataset"
nih_imgdir = os.path.join(nih_dir, "data")

# directory of the auto-formed "data" dir from COVID-NETx 
auto_dir = "C:/Users/harsh/Desktop/Python/0 - Projects/Pulmonary-Disease-Detection/0 - Datasets/autoform"
auto_datadir = os.path.join(auto_dir, "data")

# directory of the CheXpert Dataset
chexpert_dir = "C:/Users/harsh/Desktop/Python/0 - Projects/Pulmonary-Disease-Detection/0 - Datasets/CheXpert - Stanford"
# chexpert_datadir = os.path.join(chexpert_dir, 'data')

# directory of a new folder for creating "log" files
log_dir = "C:/Users/harsh/Desktop/Python/0 - Projects/Pulmonary-Disease-Detection/1 - Dataset Creator"

# new directory for the resized (downscaled) dataset
super_dataset_dir = "C:/Users/harsh/Desktop/Python/0 - Projects/Pulmonary-Disease-Detection/0 - Datasets/Super Dataset"

# parameter --> the new dataset will be resized to the following sizes
# pass one or more sizes in the format (height, width, channels)
# NOTE: Resized dataset will be named by their sizes under the `super_dataset` dir, ex: '../super_dataset_dir/128x128'
resize_to = [(256, 256), (512, 512), (720, 720)]

N_JOBS = 5  # Number of images to be resized in parallel
SEED = 777  # For reproducibility

# Setting Seeds
random.seed(SEED)

# -----------------------------------------------------------------------------
#     ================ CREATING CSVs of AVAILABLE FILES ================

# ============================== NIH DATA ==============================
nih_categories = glob(f"{nih_imgdir}/*/")

nih_labels = {
    "images_001": 'Atelectasis',
    "images_002": 'Cardiomegaly',
    "images_003": 'Effusion',
    "images_004": 'Infiltration',
    "images_005": 'Mass',
    "images_006": 'Nodule',
    "images_007": 'Pneumonia',
    "images_008": 'Pneumothorax',
    "images_009": 'Consolidation',
    "images_010": 'Edema',
    "images_011": 'Emphysema',
    "images_012": 'Fibrosis'
}

nih_log_path = f'{log_dir}/0 - nih_data.csv'

with open(nih_log_path, 'w', newline='') as nih_log:
    writer = csv.writer(nih_log)
    writer.writerow(['filename', 'label'])

    for category in nih_categories:
        for image in os.listdir(category):
            image_path = os.path.join(category, image)
            image_path = image_path.replace('\\', '/')
            image_label = nih_labels[category.split(sep='\\')[-2]].lower()
            writer.writerow([image_path, image_label])

    nih_log.close()

# ========================= AUTO-FORMED DATA ==========================

train_split = os.path.join(auto_dir, 'train_split.txt')
test_split = os.path.join(auto_dir, 'test_split.txt')

auto_train_log_path = f'{log_dir}/1 - auto_train.csv'
auto_test_log_path = f'{log_dir}/2 - auto_test.csv'

with open(train_split, 'r') as train:
    content = train.readlines()

    with open(auto_train_log_path, 'w', newline='') as auto_train:
        writer = csv.writer(auto_train)
        writer.writerow(['filename', 'label'])

        for image in content:
            label = image.split(sep=' ')[-2].lower()
            img_path = os.path.join(f"{auto_datadir}/train", image.split(sep=' ')[-3])
            img_path = img_path.replace('\\', '/')

            writer.writerow([img_path, label])

    auto_train.close()
train.close()

with open(test_split, 'r') as test:
    content = test.readlines()

    with open(auto_test_log_path, 'w', newline='') as auto_test:
        writer = csv.writer(auto_test)
        writer.writerow(['filename', 'label'])

        for image in content:
            label = image.split(sep=' ')[-2].lower()
            img_path = os.path.join(f"{auto_datadir}/test", image.split(sep=' ')[-3])
            img_path = img_path.replace('\\', '/')

            writer.writerow([img_path, label])

        auto_test.close()
    test.close()

# -----------------------------------------------------------------------------
#     ================== COMBINING THE NEWLY MADE CSVs =================

# combining images and labels to form a single huge dataframe
full_dataframe_path = f'{log_dir}/3 - full_dataframe.csv'

auto_test_csv = pd.read_csv(auto_test_log_path)
auto_train_csv = pd.read_csv(auto_train_log_path)
nih_data_csv = pd.read_csv(nih_log_path)

data_to_combine = [auto_test_csv, auto_train_csv, nih_data_csv]

with open(full_dataframe_path, 'w', newline='') as full_dataframe:
    writer = csv.writer(full_dataframe)
    writer.writerow(['filename', 'label'])

    for data in data_to_combine:
        for i, image in enumerate(data['filename']):
            writer.writerow([image, data['label'][i]])

    full_dataframe.close()


# -----------------------------------------------------------------------------
#     ======== DOWNSCALING ALL IMAGES TO MAKE A SMALLER DATASET =========
# Note: All images are downscaled AND converted to grayscale

full_dataframe = pd.read_csv(full_dataframe_path)

splits = ['train', 'test']

labels = ['covid-19', 'normal', 'atelectasis', 'cardiomegaly', 'effusion', 'infiltration',
          'mass', 'nodule', 'pneumonia', 'pneumothorax', 'consolidation', 'edema',
          'emphysema', 'fibrosis']

k_covid, k_except_covid = 500, 1000

group_by_labels = full_dataframe.groupby('label', axis=0)

start_time = time.time()

for resolution in resize_to:
    resolution_subdir = os.path.join(super_dataset_dir, f'{resolution[0]}x{resolution[0]}')
    os.mkdir(resolution_subdir)

    for label in labels:
        train_sub_path = os.path.join(resolution_subdir, 'train')
        test_sub_path = os.path.join(resolution_subdir, 'test')

        if os.path.exists(train_sub_path) and os.path.exists(test_sub_path):
            pass
        else:
            os.mkdir(train_sub_path)
            os.mkdir(test_sub_path)

        label_train_subdir = os.path.join(train_sub_path, label)
        label_test_subdir = os.path.join(test_sub_path, label)

        if os.path.exists(label_train_subdir) and os.path.exists(label_test_subdir):
            pass
        else:
            os.mkdir(label_train_subdir)
            os.mkdir(label_test_subdir)

        group_from_df = pd.DataFrame(group_by_labels.get_group(label))
        group_from_df.set_index(np.arange(len(group_from_df['filename'])), inplace=True)

        filenames = group_from_df['filename'].values.tolist()
        random.shuffle(filenames)

        if label == 'covid-19':
            train_set, test_set = filenames[:k_covid], filenames[k_covid:]
            filenames = [train_set, test_set]
        elif label != 'covid-19':
            train_set, test_set = filenames[:k_except_covid], filenames[k_except_covid:k_except_covid+500]
            filenames = [train_set, test_set]
        else:
            pass

        for i, file_split in enumerate(filenames):
            if i == 0:
                save_path = label_train_subdir
            else:
                save_path = label_test_subdir

            def resize(image):
                img = cv2.imread(image)
                resized_image = cv2.resize(img, resolution, interpolation=cv2.INTER_LINEAR)

                destination_path = f"{save_path}/{image.split('/')[-1]}"
                cv2.imwrite(destination_path, resized_image)

            def resize_and_flip(image):
                img = cv2.imread(image)
                resized_image = cv2.resize(img, resolution, interpolation=cv2.INTER_LINEAR)
                flipped_image = cv2.flip(resized_image, 1)

                resized_destination_path = f"{save_path}/{image.split('/')[-1]}"
                flipped_destination_path = f"{save_path}/(flipped){image.split('/')[-1]}"

                cv2.imwrite(resized_destination_path, resized_image)
                cv2.imwrite(flipped_destination_path, flipped_image)

            if label != 'covid-19':
                Parallel(n_jobs=N_JOBS, verbose=10)(
                    delayed(resize)(f) for f in file_split
                )

            elif label == 'covid-19':
                Parallel(n_jobs=N_JOBS, verbose=10)(
                    delayed(resize_and_flip)(f) for f in file_split
                )
            else:
                pass


stop_time = time.time()

print(f"total time taken: {stop_time - start_time}")
