# coding:utf-8

import sys
import os
import cv2
import yaml


class Yolo:
    def __init__(self):
        self.id = 25

    def detect(self):
        detections = []
        for i in range(10):
            detection = []
            for j in range(5):
                detection.append(j)
            detections.append(detection)
        return detections


class com:
    def __init__(self):
        self.id = 6

    def multiply(self, a, b):
        print(self.id, 'multiply')
        return a * b

    def add(self, a=6, b=6):
        print(self.id, 'add')
        return a + b

    def show(self, a=10):
        print(self.id, 'show')
        return a

    def name(self):
        print(self.id, 'name')


def main():
    print('Hello World python')


def hello(in_):
    print('Hello World python', in_)


def multiply(a=6, b=6):
    print('multiply')
    return a * b


if __name__ == 'main':
    main()
