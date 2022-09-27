# -*- coding: UTF-8 -*-
'''
@Project ：CRNN
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch
from torch import nn
from crnn import Crnn
from configure import config as cfg
from _utils.generate import Generator

if __name__ == '__main__':

    CRnn = Crnn(device=cfg.device,
                weight_decay=cfg.weight_decay,
                learning_rate=cfg.learning_rate,
                resume_train=cfg.load_ckpt,
                ckpt_path=cfg.ckpt_path + ".\\模型文件")

    gen = Generator(root_path=cfg.root_path,
                    training_path=cfg.training_path,
                    validate_path=cfg.validate_path,
                    cropped_size=cfg.cropped_size,
                    batch_size=cfg.batch_size)

    train_func = gen.generate(training=True)
    validate_func = gen.generate(training=False)

    for epoch in range(cfg.Epoches):

        for i in range(gen.get_train_len()):
            sources, targets = next(train_func)
            CRnn.train(sources, targets)
            if not i % cfg.per_sample_interval and i:
                CRnn.generate_sample(sources, targets, i)

        print('Epoch{:0>3d} train loss is {:.5f} train acc is {:.3f}'.format(epoch+1,
                                                                             CRnn.train_loss / gen.get_train_len(),
                                                                             CRnn.train_acc / gen.get_train_len()*100))
        torch.save({'state_dict': CRnn.model.state_dict(),
                    'loss': CRnn.train_loss / gen.get_train_len()},
                   cfg.ckpt_path + '\\Epoch{:0>3d}_train_loss{:.5f}.pth.tar'.format(
                       epoch + 1, CRnn.train_loss / gen.get_train_len()))
        CRnn.train_loss = 0
        CRnn.train_acc = 0

        for i in range(gen.get_val_len()):
            sources, targets = next(validate_func)
            CRnn.validate(sources, targets)

        print('Epoch{:0>3d} validate loss is {:.5f} validate acc is {:.3f}'.format(epoch+1,
                                                                                   CRnn.val_loss / gen.get_val_len(),
                                                                                   CRnn.val_acc / gen.get_val_len()*100))

        CRnn.val_loss = 0
        CRnn.val_acc = 0