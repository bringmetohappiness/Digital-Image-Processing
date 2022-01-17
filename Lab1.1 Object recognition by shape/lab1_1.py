#!/usr/bin/env python3
"""Здесь определён скрипт, выполняющий ЛР1."""
import os

import fruit_classifier

TRAIN_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'train')
TEST_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'test')


def main():
    classifier = fruit_classifier.FruitClassifier()
    classifier.train(TRAIN_PATH, (0.1, 0.6, 0.1), (10, 60, 10), (10, 100, 10))
    classifier.test(TEST_PATH)


if __name__ == '__main__':
    main()
