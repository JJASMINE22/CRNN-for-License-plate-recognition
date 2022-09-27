# -*- coding: UTF-8 -*-
'''
@Project ：CRNN
@File    ：utils.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import io
import os
import re
import time
import numpy as np
import config as cfg
from PIL import Image, ImageFont, ImageDraw
from configure import config as cfg
from configure.chars import characters

def letterbox(image, target_size:tuple):
    """
    add gray bar to image
    """
    iw, ih = image.size
    w, h = target_size

    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    dx = (w-nw)//2
    dy = (h-nh)//2

    image = image.resize((nw, nh), Image.BICUBIC)
    letterbox_image = Image.new('RGB', (w, h), (128, 128, 128))
    letterbox_image.paste(image, (dx, dy))

    return letterbox_image


def remove_duplicate(logits, target_size:int):
    """
    remove duplicate chars between '-'
    """
    predictions = list()
    for k, logit in enumerate(logits):
        forbid_index = list()
        total_index = np.arange(logit.__len__())
        mask_index = total_index[np.equal(logit, 0)]
        for i in range(len(mask_index)):
            if not i:
                if mask_index[i] - i > 2:
                    for j in range(mask_index[i]):
                        if np.equal(logit[j], logit[j + 1]):
                            forbid_index.append(j)
            else:
                if mask_index[i] - mask_index[i-1] > 2:
                    for j in range(mask_index[i-1] + 1, mask_index[i]):
                        if np.equal(logit[j], logit[j + 1]):
                            forbid_index.append(j)
        forbid_index.extend(mask_index)
        prediction = np.delete(np.array(logit), obj=forbid_index)
        # prevent insufficient chars in the initial iter
        try:
            if np.equal(prediction[0], prediction[1]):
                predictions.append(prediction[1:target_size + 1])
            else:
                predictions.append(prediction[:target_size])
        except IndexError:
            predictions.append(prediction[:target_size])

    return predictions


def image_mark(image, logit, batch):

    text = str()
    for _ in np.squeeze(logit):
        text += characters[_]

    font = ImageFont.truetype(font=cfg.font_path,
                              size=np.floor(3e-1 * image.size[1] + 0.5).astype('int'))
    draw = ImageDraw.Draw(image)

    text_size = draw.textsize(text, font)
    # text = text.encode('utf-8')
    draw.text(np.array([image.size[0] // 3, image.size[1] // 2]),
              text, fill=(255, 0, 0), font=font)

    del draw

    image.save(cfg.sample_path.format(batch), quality=95, subsampling=0)
