"""Здесь определён простой классификатор, распознающий фрукты по форме."""
import math
import os

import numpy as np
from skimage import io
from skimage import filters

_EPSILON = 1e-8
_BLACK_PIXEL = 0
_WHITE_PIXEL = 255
_CENTER = 49.5


class FruitClassifier:
    def __init__(self):
        self.fruit1_name = None
        self.fruit2_name = None

        self.bw_threshold = None
        self.radius = None
        self.variance_threshold = None

    def train(self, path, bw_param, radius_param, variance_param):
        """Перебором находит лучшие параметры для классификации фруктов, используя обучающий набор.

        Args:
            path: путь, по которому лежит обучающий набор данных.
            bw_param: параметры перебора границы перевода изображения в чёрно-белый формат.
            radius_param: параметры перебора радиуса, в котором не будут учитываться точки.
            variance_param: параметры перебора границы дисперсии при предсказании.
        """
        self.fruit1_name = os.listdir(path)[0]  # Ginger Root
        self.fruit2_name = os.listdir(path)[1]  # Physalis

        best_accuracy = 0
        best_bw_threshold = None
        best_radius = None
        best_variance_threshold = None

        bw_threshold = bw_param[0]
        while best_accuracy != 1 and abs(bw_threshold - bw_param[1]) > _EPSILON:
            radius = radius_param[0]
            while best_accuracy != 1 and radius != radius_param[1]:
                variance_threshold = variance_param[0]
                while best_accuracy != 1 and variance_threshold != variance_param[1]:
                    tp, fn, tn, fp = self._tp_fn_tn_fp(
                        path, bw_threshold, radius, variance_threshold
                    )
                    current_accuracy = self._accuracy(tp, fn, tn, fp)
                    if best_accuracy < current_accuracy:
                        best_accuracy = current_accuracy
                        best_bw_threshold = bw_threshold
                        best_radius = radius
                        best_variance_threshold = variance_threshold
                    if abs(best_accuracy - 1.0) < _EPSILON:
                        best_accuracy = 1
                    variance_threshold += variance_param[2]
                radius += radius_param[2]
            bw_threshold += bw_param[2]

        self.bw_threshold = best_bw_threshold
        self.radius = best_radius
        self.variance_threshold = best_variance_threshold

    @staticmethod
    def _accuracy(tp, fn, tn, fp):
        """Возвращает точность."""
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return accuracy

    @staticmethod
    def _precision(tp, fp):
        """Возвращает прецизионность."""
        precision = tp / (tp + fp)
        return precision

    @staticmethod
    def _negative_predictive_value(tn, fn):
        """Возвращает отрицательное прогностическое значение."""
        negative_predictive_value = tn / (tn + fn)
        return negative_predictive_value

    @staticmethod
    def _sensitivity(tp, fn):
        """Возвращает чувствительность."""
        sensitivity = tp / (tp + fn)
        return sensitivity

    @staticmethod
    def _specificity(tn, fp):
        """Возвращает специфичность."""
        specificity = tn / (tn + fp)
        return specificity

    def _tp_fn_tn_fp(self, path, bw_threshold, radius, variance_threshold):
        """Возвращает TruePositive, FalseNegative, TrueNegative и FalsePositive."""
        tp, fn = self._get_tp_fn_or_tn_fp(
            path, self.fruit1_name, bw_threshold, radius, variance_threshold
        )
        tn, fp = self._get_tp_fn_or_tn_fp(
            path, self.fruit2_name, bw_threshold, radius, variance_threshold
        )
        return tp, fn, tn, fp

    def _get_tp_fn_or_tn_fp(self, path, fruit_name, threshold_get_bin, radius, variance_threshold):
        """Возвращает TruePositive и FalseNegatives или TrueNegatives и FalsePositives
        в зависимости от предоставленного фрукта.

        Если fruit_name == self.fruit1_name вернёт TruePositives и FalseNegatives.
        Если fruit_name == self.fruit2_name вернёт TrueNegatives и FalsePositives.
        """
        filenames = os.listdir(os.path.join(path, fruit_name))
        tp_or_tn = 0
        fn_or_fp = 0
        for filename in filenames:
            if self.predict(
                    os.path.join(path, fruit_name, filename),
                    threshold_get_bin,
                    variance_threshold,
                    radius=radius,
            ) == fruit_name:
                tp_or_tn += 1
            else:
                fn_or_fp += 1
        return tp_or_tn, fn_or_fp

    def predict(self, path, bw_threshold, variance_threshold, radius=0):
        """Предсказывает, что за фрукт на изображении.

        Args:
            path: путь, по которому лежит изображение.
            bw_threshold: значение порога яркости при переводе изображения в чёрно-белый формат.
            radius: радиус окружности в центре изображения, в котором не нужно считать расстояния.
            variance_threshold: значение порога вероятности для предсказания фрукта.

        Returns:
            predict: предсказание - название фрукта.
        """
        image = io.imread(path, as_gray=True)
        image = filters.roberts(image)
        image = self.black_whited(image, bw_threshold)
        distances = self._get_distances(image, radius=radius)
        variance = np.var(distances)
        predict = self.fruit1_name if variance > variance_threshold else self.fruit2_name
        return predict

    @staticmethod
    def black_whited(image, bw_threshold):
        """Переводит изображение в оттенках серого в чёрно-белый формат.

        Args:
            image: изображение в оттенках серого.
            bw_threshold: пороговое значение.

        Returns:
            bw_image: чёрно-белое изображение.
        """
        bw_image = []
        for line in image:
            bw_line = []
            for pixel in line:
                if pixel <= bw_threshold:
                    bw_line.append(_WHITE_PIXEL)
                else:
                    bw_line.append(_BLACK_PIXEL)
            bw_image.append(bw_line)
        bw_image = np.array(bw_image)
        return bw_image

    @staticmethod
    def _get_distances(image, radius=0):
        """Возвращает список расстояний между белыми пикселями и центром чёрно-белого изображения.

        Возвращает список расстояний между белыми пикселями и центром чёрно-белого изображения,
        кроме тех пикселей, что лежат в центральной окружности.

        Args:
            image: чёрно-белое изображение.
            radius: радиус окружности в центре изображения, в котором не нужно считать расстояния.

        Returns:
            distances: список расстояний между белыми пикселями и центром чёрно-белого изображения.
        """
        distances = []
        for i in range(100):
            for j in range(100):
                if image[i][j] == _BLACK_PIXEL:
                    distance = math.sqrt((i - _CENTER)**2 + (j - _CENTER)**2)
                    if radius and distance >= radius:
                        distances.append(distance)
        distances = np.array(distances)
        return distances

    def test(self, path):
        """Подсчитывает оценки классификации, используя найденные во время обучения на тестовом
        наборе параметры.

        Args:
            path: путь, по которому лежит тестовый набор данных.
        """
        tp, fn, tn, fp = self._tp_fn_tn_fp(
            path, self.bw_threshold, self.radius, self.variance_threshold
        )
        accuracy = self._accuracy(path, self.bw_threshold, self.radius, self.variance_threshold)
        precision = self._precision(tp, fp)
        negative_predictive_value = self._negative_predictive_value(tn, fn)
        sensitivity = self._sensitivity(tp, fn)
        specificity = self._specificity(tn, fp)
        print(
            f'accuracy = {accuracy:4.2%}, precision = {precision:4.2%}'
            f'negative predictive value = {negative_predictive_value:4.2%}'
            f'sensitivity = {sensitivity:4.2%}'
            f'specificity = {specificity:4.2%}'
        )
