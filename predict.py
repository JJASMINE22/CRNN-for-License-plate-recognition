# -*- coding: UTF-8 -*-
'''
@Project ：CRNN
@File    ：train.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
import os
import time
import torch
import numpy as np
import config as cfg
from torch import nn
from PIL import Image
from net.networks import CRNN
from _utils.generate import Generator
from _utils.utils import image_mark, remove_duplicate

if __name__ == '__main__':

    model = CRNN()
    if cfg.device:
        model = model.to(cfg.device)

    try:
        ckpt = torch.load(os.path.join(cfg.ckpt_path, '模型文件'))
        model.load_state_dict(ckpt['state_dict'])
        print("model successfully loaded, ctc loss {:.3f}".format(ckpt['loss']))
    except:
        raise ("please enter the right params path")

    model = model.eval()

    gen = Generator(root_path=cfg.root_path,
                    training_path=cfg.training_path,
                    validate_path=cfg.validate_path,
                    cropped_size=cfg.cropped_size,
                    batch_size=cfg.batch_size)

    validate_func = gen.generate(training=False)

    for i in range(gen.get_val_len()):
        sources, targets = next(validate_func)
        if cfg.device:
            sources = torch.tensor(sources, dtype=torch.float).to(cfg.device)

        random_idx = np.random.choice(sources.shape[0], 1)

        source = sources[random_idx]
        logit = model(source)

        logit = logit.detach().transpose(0, 1).argmax(dim=-1).cpu().numpy()
        source = source.squeeze().permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(np.uint8((source + 1) * 127.5))

        logit = remove_duplicate(logit, cfg.target_size)

        image_mark(image, logit, 1)

        time.sleep(3)