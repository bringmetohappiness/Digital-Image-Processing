"""Здесь написан скрипт, рисующий картинки для отчёта."""
import os

import matplotlib.pyplot as plt
from skimage import color

import fruit_classifier

GINGER_ROOT_TRAIN = os.path.join(os.path.dirname(__file__), 'dataset', 'train', 'Ginger Root')
PHYSALIS_TRAIN = os.path.join(os.path.dirname(__file__), 'dataset', 'train', 'Physalis')


def show_fruit(path):
    """Показывает 9 изображений из каталога."""
    file_names = os.listdir(path)
    fig, axes = plt.subplots(3, 3)
    axes = axes.flatten()
    for ax, file_name in zip(axes, file_names[:9]):
        image = plt.imread(os.path.join(path, file_name))
        ax.imshow(image)
        ax.set_axis_off()
    fruit_name = os.path.basename(path)
    fig.suptitle(f'Некоторые из {fruit_name}')
    plt.show()


def first_image(path):
    """Возвращает первое изображение в директории."""
    filename = os.listdir(path)[0]
    image = plt.imread(os.path.join(path, filename))
    return image


def show_processing(image):
    """Показывает обработку изображения."""
    gray_scale_image = color.rgb2gray(image)
    black_white_image = fruit_classifier.FruitClassifier.black_whited(gray_scale_image, 0.4)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(image)
    axes[0].set_title('Исходное изображение')
    axes[1].imshow(gray_scale_image, cmap=plt.cm.gray)
    axes[1].set_title('Изображение в оттенках серого')
    axes[2].imshow(black_white_image, cmap=plt.cm.gray)
    axes[2].set_title('Чёрно-белое изображение')
    fig.suptitle('Обработка изображения', fontsize=16)
    plt.show()


def main():
    show_fruit(GINGER_ROOT_TRAIN)
    show_fruit(PHYSALIS_TRAIN)

    image = first_image(GINGER_ROOT_TRAIN)
    show_processing(image)


if __name__ == '__main__':
    main()
