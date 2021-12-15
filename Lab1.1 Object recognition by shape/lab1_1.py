#!/usr/bin/env python3
import math
import os

import numpy as np
from skimage import io, filters


TRAIN_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'train')
TEST_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'test')
FRUIT1_NAME = os.listdir(TRAIN_PATH)[0]  # Ginger Root
FRUIT2_NAME = os.listdir(TRAIN_PATH)[1]  # Physalis
EPSILON = 1e-8
BLACK_PIXEL = 0
WHITE_PIXEL = 255
CENTER = 49.5


def get_bw_image(image, bw_threshold):
    """Returns black-white image after thresholding."""
    bw_image = []
    for line in image:
        bw_line = []
        for pixel in line:
            if pixel <= bw_threshold:
                bw_line.append(WHITE_PIXEL)
            else:
                bw_line.append(BLACK_PIXEL)
        bw_image.append(bw_line)
    return np.array(bw_image)


def get_distances(image, radius=0):
    """
    Returns an array of distances between WHITE pixels and a CENTER of an image.

    Returns an array of distances between WHITE pixels and a CENTER of an image,
    except for those pixels that fall within the circle in the center.
    radius - of the circle in which distances are not counted.
    """
    distances = []
    for i in range(100):
        for j in range(100):
            if image[i][j] == BLACK_PIXEL:
                distance = math.sqrt(pow(i - CENTER, 2) + pow(j - CENTER, 2))
                if radius and distance >= radius:
                    distances.append(distance)
    return np.array(distances)


def get_predict(path, bw_threshold, variance_threshold, radius=0):
    """
    Gives a prediction from the input image which fruit is in the image.

    Has 3 more additional arguments:
    bw_threshold - value of the brightness threshold when
    creating a black-and-white image;
    radius - radius of the circle in the center, in which the distance
    is not estimated;
    variance_threshold - variance threshold value for fruit prediction.
    """
    image = io.imread(path, as_gray=True)
    image = filters.roberts(image)
    image = get_bw_image(image, bw_threshold)
    distances = get_distances(image, radius=radius)
    variance = np.var(distances)
    predict = FRUIT1_NAME if (variance > variance_threshold) else FRUIT2_NAME
    return predict


def get_tp_fn_or_tn_fp(path, fruit_name, threshold_get_bin, radius, variance_threshold):
    """
    Returns True Positives and False Negatives or True Negatives and
    False Positives depending on the supplied fruit name.

    Returns True Positives and False Negatives if fruit_name = FRUIT1_NAME
        and True Negatives and False Positives if fruit_name = FRUIT2_NAME.
    """
    filenames = os.listdir(os.path.join(path, fruit_name))
    true_positives_or_true_negatives = 0
    false_negatives_or_false_positives = 0
    for filename in filenames:
        if get_predict(
            os.path.join(path, fruit_name, filename),
            threshold_get_bin,
            variance_threshold,
            radius=radius
        ) == fruit_name:
            true_positives_or_true_negatives += 1
        else:
            false_negatives_or_false_positives += 1
    return true_positives_or_true_negatives, false_negatives_or_false_positives


def get_tp_fn_tn_fp(path, threshold_get_bin, radius, variance_threshold):
    """Returns True Positive, False Negative, True Negative, and False Positive as a single tuple.
    """
    true_positives, false_negatives = get_tp_fn_or_tn_fp(
        path,
        FRUIT1_NAME,
        threshold_get_bin,
        radius,
        variance_threshold
    )
    true_negatives, false_positives = get_tp_fn_or_tn_fp(
        path,
        FRUIT2_NAME,
        threshold_get_bin,
        radius,
        variance_threshold
    )
    return true_positives, false_negatives, true_negatives, false_positives


def get_accuracy(true_positives, true_negatives, false_positives, false_negatives):
    """Returns accuracy."""
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives +
                                                    false_positives + false_negatives)
    return accuracy


def get_precision(true_positives, false_positives):
    """Returns precision."""
    precision = true_positives / (true_positives + false_positives)
    return precision


def get_negative_predictive_value(true_negatives, false_negatives):
    """Returns negative predictive value."""
    negative_predictive_value = true_negatives / (true_negatives + false_negatives)
    return negative_predictive_value


def get_sensitivity(true_positives, false_negatives):
    """Returns sensitivity."""
    sensitivity = true_positives / (true_positives + false_negatives)
    return sensitivity


def get_specificity(true_negatives, false_positives):
    """Returns specificity."""
    specificity = true_negatives / (true_negatives + false_positives)
    return specificity


def train(path):
    """Sorting out to find the best parameters for fruit classification using a training set."""
    table_caption = (
        '| threshold_get_bin |'
        ' radius |'
        ' variance_threshold |'
        '| current accuracy |'
        ' best accuracy |'
    )
    line_length = len(table_caption)
    print('_' * line_length)
    print(table_caption)
    print('=' * line_length)

    best_accuracy = None
    threshold_get_bin = 0.1  # Fine tuning
    while best_accuracy != 1 and abs(threshold_get_bin - 0.6) > EPSILON:
        radius = 10  # Fine tuning
        while best_accuracy != 1 and radius != 60:
            variance_threshold = 10  # Fine tuning
            while best_accuracy != 1 and variance_threshold != 100:
                true_positives, false_negatives, true_negatives, false_positives = get_tp_fn_tn_fp(
                    path, threshold_get_bin,
                    radius, variance_threshold
                )
                current_accuracy = get_accuracy(
                    true_positives,
                    true_negatives,
                    false_positives,
                    false_negatives
                )
                if best_accuracy is None or best_accuracy < current_accuracy:
                    best_accuracy = current_accuracy
                    best_threshold_get_bin = threshold_get_bin
                    best_radius = radius
                    best_variance_threshold = variance_threshold
                if abs(best_accuracy - 1.0) < EPSILON:
                    best_accuracy = 1
                print(f'|        {threshold_get_bin:3.2}        |'
                      f'   {radius:2}   |'
                      f'         {variance_threshold:2}         |'
                      f'|      {current_accuracy:4.2%}      |'
                      f'     {best_accuracy:4.2%}    |'
                      )
                variance_threshold += 10
            print('-' * line_length)
            radius += 10
        print('-' * line_length)
        threshold_get_bin += 0.1

    print(
        f'best threshold_get_bin = {best_threshold_get_bin}'
        f'best radius = {best_radius}'
        f'best variance_threshold = {best_variance_threshold}'
        )
    return best_threshold_get_bin, best_radius, best_variance_threshold


def test(best_threshold_get_bin, best_radius, best_variance_threshold):
    """Calculates the scores of the developed method with the found best parameters on the
    test dataset.
    """
    true_positives, false_negatives, true_negatives, false_positives = get_tp_fn_tn_fp(
        TEST_PATH, best_threshold_get_bin,
        best_radius, best_variance_threshold
    )
    accuracy = get_accuracy(true_positives, true_negatives, false_positives, false_negatives)
    precision = get_precision(true_positives, false_positives)
    negative_predictive_value = get_negative_predictive_value(true_negatives, false_negatives)
    sensitivity = get_sensitivity(true_positives, false_negatives)
    specificity = get_specificity(true_negatives, false_positives)
    print(f'accuracy = {accuracy:4.2%}, precision = {precision:4.2%}'
          f'negative predictive value = {negative_predictive_value:4.2%}'
          f'sensitivity = {sensitivity:4.2%}'
          f'specificity = {specificity:4.2%}'
          )


def main():
    best_threshold_get_bin, best_radius, best_variance_threshold = train(TRAIN_PATH)
    test(best_threshold_get_bin, best_radius, best_variance_threshold)


if __name__ == '__main__':
    main()
