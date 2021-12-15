#!/usr/bin/env python3
import os
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np


TRAIN_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'train')
TEST_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'test')
CHANNEL1 = 0
CHANNEL2 = 1
CHANNEL3 = 2
H_MAX = 180
L_MAX = 256
S_MAX = 256
MASK = None
H_RANGE = [0, H_MAX]
L_RANGE = [0, L_MAX]
S_RANGE = [0, S_MAX]
# Fine tuning by hands!
TRAIN_SETTINGS = dict(dp=2, minDist=350, param1=650, param2=100, minRadius=175, maxRadius=225)
# Fine tuning by hands!
TEST_SETTINGS = dict(dp=2, minDist=300, param1=500, param2=100, minRadius=130, maxRadius=140)
GREEN_COLOR = (0, 1.0, 0)
WHITE_COLOR = (1.0, 1.0, 1.0)


def circles_check(path):
    """
    Used to set up arguments in a function cv2.HoughCircles.

    path - to the directory with images.
    """
    filenames = os.listdir(path)
    for filename in filenames:
        image = plt.imread(os.path.join(path, filename))
        grayscale_image = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        # Fine tuning by hands!
        circles = cv2.HoughCircles(
            grayscale_image,
            cv2.HOUGH_GRADIENT,
            dp=2,
            minDist=300,
            param1=500,
            param2=100,
            minRadius=130,
            maxRadius=140
        )
        ax = plt.subplot()
        if circles is not None:
            for circle in circles[0]:
                x, y, r = map(int, circle)
                draw_circle = patches.Circle(
                    (x, y),
                    radius=r,
                    color=GREEN_COLOR,
                    linewidth=3,
                    fill=False
                )
                ax.imshow(image)
                ax.add_patch(draw_circle)
        else:
            print('No circles found.')

        plt.show()


def get_circles(image, settings):
    """Returns a list of the circles found in the image.

    image - the image on which the circles are searched.
    settings - a named argument dictionary for the HoughCircles function.
    HoughCircles: https://docs.opencv.org/4.5.1/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(grayscale_image, cv2.HOUGH_GRADIENT, **settings)
    return circles[0]


def get_intermediate_accumulator(image):
    """Returns the accumulator histogram for the cropped parts of the image with circles."""
    intermediate_accumulator = np.zeros((H_MAX, L_MAX, S_MAX))
    circles = get_circles(image, TRAIN_SETTINGS)
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    for circle in circles:
        x, y, r = map(int, circle)
        crop = hls_image[y-r:y+r, x-r:x+r, :]
        hist = cv2.calcHist(
            [crop],
            [CHANNEL1, CHANNEL2, CHANNEL3],
            MASK,
            [H_MAX, L_MAX, S_MAX],
            H_RANGE + L_RANGE + S_RANGE,
            accumulate=False
        )
        intermediate_accumulator += hist
    return intermediate_accumulator


def get_mean_hist(path, label):
    """Returns the mean histogram for the label."""
    filenames = os.listdir(os.path.join(path, label))
    accumulator = np.zeros((H_MAX, L_MAX, S_MAX))
    for filename in filenames:
        image = cv2.imread(os.path.join(path, label, filename))
        intermediate_accumulator = get_intermediate_accumulator(image)
        accumulator += intermediate_accumulator
    mean_hist = accumulator / len(filenames)
    mean_hist = mean_hist.astype('float32')
    return mean_hist


def train(path):
    """
    Returns a dictionary where the key=label, value=mean histogram.

    path - path to train dataset.
    """
    labels = os.listdir(path)
    some_dict = {}
    for label in labels:
        some_dict[label] = get_mean_hist(path, label)
    return some_dict


def get_predict(hist, train_results):
    """
    Returns predicted label for input hisogram.

    train_results - dictionary where the key=label, value=mean histogram.
    """
    some_dict = train_results.copy()

    for label, label_hist in some_dict.items():
        compare = cv2.compareHist(hist, label_hist, cv2.HISTCMP_CORREL)
        some_dict[label] = compare

    predict = None
    max_compare = None
    for label, compare in some_dict.items():
        if max_compare is None or compare > max_compare:
            max_compare = compare
            predict = label
    return predict


def test(path, train_results):
    """
    Shows test images with found circles and predicted labels for them.

    path - path to test dataset.
    train_results - dictionary where the key=label, value=mean histogram.
    """
    _, axes = plt.subplots(5, 4, figsize=(8, 8))
    axes = axes.flatten()
    filenames = os.listdir(path)
    for ax, filename in zip(axes, filenames):
        bgr_image = cv2.imread(os.path.join(path, filename))
        rgb_image = plt.imread(os.path.join(path, filename))
        circles = get_circles(bgr_image, TEST_SETTINGS)
        hls_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HLS)
        for circle in circles:
            x, y, r = map(int, circle)
            crop = hls_image[y-r:y+r, x-r:x+r, :]
            hist = cv2.calcHist(
                [crop],
                [CHANNEL1, CHANNEL2, CHANNEL3],
                MASK,
                [H_MAX, L_MAX, S_MAX],
                H_RANGE + L_RANGE + S_RANGE,
                accumulate=False
            )
            predict = get_predict(hist, train_results)

            draw_circle = patches.Circle(
                (x, y),
                radius=r,
                color=GREEN_COLOR,
                linewidth=3,
                fill=False
            )
            ax.imshow(rgb_image)
            ax.add_patch(draw_circle)
            ax.text(x, y, predict, fontsize='xx-large', backgroundcolor=WHITE_COLOR)
    plt.show()


def main():
    train_results = train(TRAIN_PATH)
    test(TEST_PATH, train_results)


if __name__ == '__main__':
    main()
