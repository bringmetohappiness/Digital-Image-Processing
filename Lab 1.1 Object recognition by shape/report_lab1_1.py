"""Report for lab1 in Digital Image Processing."""

import os
from skimage import color
import matplotlib.pyplot as plt
import lab1_1


GINGER_ROOT_TRAIN = os.path.join(lab1_1.TRAIN_PATH, lab1_1.FRUIT1_NAME)
PHYSALIS_TRAIN = os.path.join(lab1_1.TRAIN_PATH, lab1_1.FRUIT2_NAME)


def show_fruit(path):
    """Shows 9 examples of images in directory."""
    filenames = os.listdir(path)
    fig, axes = plt.subplots(3, 3)
    axes = axes.flatten()
    for ax, filename in zip(axes, filenames[:9]):
        image = plt.imread(os.path.join(path, filename))
        ax.imshow(image)
        ax.set_axis_off()
    fruit_name = os.path.basename(path)
    fig.suptitle(f'Some of {fruit_name}')
    plt.show()


def get_first_image(path):
    """Returns the first image in the directory."""
    filename = os.listdir(path)[0]
    image = plt.imread(os.path.join(path, filename))
    return image


def show_processing(image):
    """Shows Image Processing.

    The first figure is an original image.
    The second figure is grayscaled image.
    The third figure is blackwhited image.
    """
    gray_scale_image = color.rgb2gray(image)
    black_white_image = lab1_1.get_bw_image(gray_scale_image, 0.4)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[1].imshow(gray_scale_image, cmap=plt.cm.gray)
    axes[1].set_title('Grayscale Image')
    axes[2].imshow(black_white_image, cmap=plt.cm.gray)
    axes[2].set_title('Black-white Image')
    fig.suptitle('Image Processing', fontsize=16)
    plt.show()


def main():
    show_fruit(GINGER_ROOT_TRAIN)
    show_fruit(PHYSALIS_TRAIN)

    image = get_first_image(GINGER_ROOT_TRAIN)
    show_processing(image)


if __name__ == '__main__':
    main()
