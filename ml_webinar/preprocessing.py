import os
import glob
from shutil import rmtree, copyfile

from PIL import Image

import numpy as np
import tensorflow as tf

def create_first_level_subdirs(DATA_FOLDER):
    train_dir = os.path.join(DATA_FOLDER, 'train')
    val_dir = os.path.join(DATA_FOLDER, 'val')
    test_dir = os.path.join(DATA_FOLDER, 'test')

    return train_dir, val_dir, test_dir


def create_empty_class_subdirs(train_dir, val_dir, test_dir, class_categories):
    # create empty split folder
    for selected_dir in [train_dir, val_dir, test_dir]:
        if not os.path.exists(selected_dir):
            os.mkdir(selected_dir)

    # create subdirectories
    for class_category in class_categories:
        class_train_dir = os.path.join(train_dir, class_category)
        class_val_dir = os.path.join(val_dir, class_category)
        class_test_dir = os.path.join(test_dir, class_category)

        for class_dir in [class_train_dir, class_val_dir, class_test_dir]:
            # Start with an empty directory
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)

            else:
                rmtree(class_dir)
                os.mkdir(class_dir)


def fill_class_subdirs(DATA_FOLDER, train_dir, val_dir, test_dir, class_categories,
                       train_ratio, val_ratio, flip, grayscale):
    # fill subdirs
    for class_category in class_categories:
        class_train_dir = os.path.join(train_dir, class_category)
        class_val_dir = os.path.join(val_dir, class_category)
        class_test_dir = os.path.join(test_dir, class_category)

        image_paths = glob.glob(os.path.join(DATA_FOLDER, class_category, '*.jpg'))

        for image_path in image_paths:
            file_name = os.path.basename(image_path)
            file_name_flipped = f'f_{file_name}'
            file_name_gray = f'g_{file_name}'

            picture = Image.open(image_path)
            array = np.asarray(picture)

            if flip:
                array_flipped = tf.image.flip_left_right(array)
                im_flipped = Image.fromarray(array_flipped.numpy())

                flipped_category = {
                    "F": "F",
                    "B": "B",
                    "FL": "FR",
                    "FR": "FL",
                    "L": "R",
                    "R": "L"}.get(class_category, class_category)
                if grayscale:
                    array_grayscale = tf.image.rgb_to_grayscale(array)
                    im_grayscale = Image.fromarray(np.repeat(array_grayscale.numpy(), 3, axis=2))

            rn = np.random.rand()
            if rn < train_ratio:
                new_path = os.path.join(class_train_dir, file_name)
                if flip:
                    flipped_path = os.path.join(DATA_FOLDER, 'train', flipped_category, file_name_flipped)
                    im_flipped.save(flipped_path)
                if grayscale:
                    grayscale_path = os.path.join(DATA_FOLDER, 'train', flipped_category, file_name_gray)
                    im_grayscale.save(grayscale_path)

            elif rn > (1 - val_ratio):
                new_path = os.path.join(class_val_dir, file_name)

            else:
                new_path = os.path.join(class_test_dir, file_name)

            copyfile(image_path, new_path)


def make_train_test(data_folder, class_dict, train_ratio, val_ratio, flip=True, grayscale=False):
    train_dir, val_dir, test_dir = create_first_level_subdirs(data_folder)
    class_categories = list(class_dict.keys())
    create_empty_class_subdirs(train_dir, val_dir, test_dir, class_categories)
    fill_class_subdirs(data_folder, train_dir, val_dir, test_dir, class_categories,
                       train_ratio, val_ratio, flip, grayscale)

# def make_generator(data_folder, phase):
#     image_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
#     return image_gen.flow_from_directory(directory=os.path.join(data_folder, phase),
#                                                      batch_size=32,
#                                                      shuffle=True,
#                                                      target_size=(IMG_PIX, IMG_PIX),
#                                                      classes=list(CLASS_DICT.keys()),
#                                                      class_mode='sparse',
#                                                      color_mode="rgb",
#                                                      seed=SEED
#                                                     )