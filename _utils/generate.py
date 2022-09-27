# -*- coding: UTF-8 -*-
'''
@Project ：CRNN
@File    ：generate.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import os
import re
import time
import numpy as np
from PIL import Image
from configure import config as cfg
from _utils.utils import letterbox


class Generator():
    def __init__(self,
                 root_path: str,
                 training_path: str,
                 validate_path: str,
                 cropped_size: tuple,
                 batch_size: int):
        self.root_path = root_path
        self.training_path = training_path
        self.validate_path = validate_path
        self.cropped_size = cropped_size
        self.batch_size = batch_size
        self.train_lines = open(self.training_path, 'rb').readlines()
        self.val_lines = open(self.validate_path, 'rb').readlines()

    def get_train_len(self):

        len_lines = self.train_lines.__len__()  # 40000
        if not len_lines % self.batch_size:
            return len_lines // self.batch_size
        else:
            return len_lines // self.batch_size + 1

    def get_val_len(self):

        len_lines = 5000  # self.val_lines.__len__()
        if not len_lines % self.batch_size:
            return len_lines // self.batch_size
        else:
            return len_lines // self.batch_size + 1

    def image_preprocess(self, image):
        image = np.array(image)
        image = image/127.5 - 1
        image = np.clip(image, -1., 1.)
        image = image.transpose([2, 0, 1])
        return image

    def generate(self, training=True):
        if training:
            lines = self.train_lines
            np.random.shuffle(lines)
            lines = lines   # [:40000]
        else:
            lines = self.val_lines
            np.random.shuffle(lines)
            lines = lines[:5000]
        while True:
            targets, sources = [], []
            for index, line in enumerate(lines):
                image_path = os.path.join(self.root_path, line.strip().decode('utf-8'))
                key_messege = image_path.split('/')[-1].split('-')

                image = Image.open(image_path)
                coordinates = key_messege[2]
                coordinates = map(int, re.sub(r"[^a-zA-Z0-9 ]", r" ", coordinates).split())
                letterbox_image = letterbox(image.crop((coordinates)), self.cropped_size)

                label = key_messege[4]
                label = map(lambda x: int(x) + 1, re.sub(r"[^a-zA-Z0-9 ]", r" ", label).split())
                label = np.array(list(label))
                label[1:] = label[1:] + cfg.mask_num

                sources.append(self.image_preprocess(letterbox_image))
                targets.append(label.tolist())

                if np.logical_or(np.equal(sources.__len__(), self.batch_size),
                                 np.equal(index, lines.__len__() - 1)):
                    annotation_sources, annotation_targets = sources.copy(), targets.copy()
                    sources.clear()
                    targets.clear()
                    yield np.array(annotation_sources), np.array(annotation_targets)
