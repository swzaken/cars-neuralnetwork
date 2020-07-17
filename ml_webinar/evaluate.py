import glob
import logging
import os


from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf


def get_images(sub_folder, data_folder, class_dict, img_pix):
    labels = []
    path = os.path.join(data_folder, sub_folder)
    images = glob.glob(os.path.join(data_folder, sub_folder, '*', '*.jpg'))
    n_images = len(images)
    logging.info('%s images in %s folder', n_images, path)

    folder_img_array = np.empty((n_images, img_pix, img_pix, 3))

    for idx, file in enumerate(images):
        picture = Image.open(file)
        vector = np.asarray(picture)
        vector_resized = tf.image.resize(vector, [img_pix, img_pix])
        folder_img_array[idx, :, :] = vector_resized

        # LABELS.append(CLASS_DICT[sub_folder]) # append the right label to the main label list
        cat = os.path.basename(os.path.dirname(file))
        labels.append(class_dict[cat]['index'])

    return folder_img_array, labels

def make_confusion_matrix(y_true, y_pred, class_dict):
    predicted = y_pred.argmax(axis=1)
    df_class = pd.DataFrame(np.c_[y_true, predicted],
                           columns=['actual', 'predicted'])
    df_class['count'] = 1
    class_matrix = pd.pivot_table(data=df_class,
                                  index='actual',
                                  columns='predicted',
                                  aggfunc='sum',
                                  values='count',
                                  fill_value=0)


    reversed_class_index = {v['index']: v['name'] for k, v in class_dict.items()}

    for key, _ in reversed_class_index.items():
        if key not in class_matrix.columns:
            class_matrix[key] = 0

    class_matrix = class_matrix.reindex(sorted(class_matrix.columns), axis=1)

    class_matrix.index = class_matrix.index.map(reversed_class_index)
    class_matrix.columns = class_matrix.columns.map(reversed_class_index)

    return class_matrix

def get_accuracy(df_confusion):
    return np.trace(df_confusion) / df_confusion.sum().sum()

