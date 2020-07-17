import matplotlib.pyplot as plt

import numpy as np


def plot_eval_image(predictions_array, true_label, img, class_dict):
    reversed_class_index = {v['index']: v['name'] for k, v in class_dict.items()}

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img / 256, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(reversed_class_index[predicted_label],
                                         100 * np.max(predictions_array),
                                         reversed_class_index[true_label]),
               color=color)


def plot_image_prediction(predictions_array, true_label):
    n_predictions = len(predictions_array)
    plt.grid(False)
    plt.xticks(range(n_predictions))
    plt.yticks([])
    thisplot = plt.bar(range(n_predictions), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def plot_image_and_predictions(test_image, test_label, array_predictions, class_dict):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_eval_image(array_predictions, test_label, test_image, class_dict)

    plt.subplot(1, 2, 2)
    plot_image_prediction(array_predictions, test_label)


def plot_multiple_images_and_predictions(predictions, test_labels, test_images, CLASS_DICT,
                                         num_rows=3, num_cols=3):
    num_images = num_rows * num_cols

    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        idx = np.random.choice(range(len(predictions)), 1)[0]

        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_eval_image(predictions[idx], test_labels[idx], test_images[idx], CLASS_DICT)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_image_prediction(predictions[idx], test_labels[idx])
    plt.tight_layout()
    plt.show()




