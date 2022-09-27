# -*- coding: UTF-8 -*-
'''
@Project ：CRNN
@File    ：crnn.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch
import numpy as np
import config as cfg
from torch import nn
from PIL import Image
from net.networks import CRNN
from _utils.utils import remove_duplicate, image_mark

class Crnn:
    def __init__(self,
                 device,
                 weight_decay,
                 learning_rate,
                 resume_train,
                 ckpt_path):
        """
        :param device: cuda or cpu
        :param weight_decay: weight of l2 regularization
        """

        self.device = device

        self.model = CRNN()
        if self.device:
            self.model = self.model.to(self.device)

        if resume_train:
            try:
                ckpt = torch.load(ckpt_path)
                self.model.load_state_dict(ckpt['state_dict'])
                print("model successfully loaded, ctc loss {:.3f}".format(ckpt['loss']))
            except:
                raise ("please enter the right params path")

        self.weight_decay = weight_decay

        self.loss = nn.CTCLoss(blank=0)
        self.optimizer = torch.optim.Adam(lr=learning_rate,
                                          params=self.model.parameters())

        self.train_loss, self.val_loss = 0, 0
        self.train_acc, self.val_acc = 0, 0

    def train(self, sources, targets):
        if self.device:
            sources = torch.tensor(sources, dtype=torch.float).to(self.device)
            targets = torch.tensor(targets, dtype=torch.long).to(self.device)
            source_lengths = torch.full(size=(sources.size(0),),
                                        fill_value=cfg.logit_size, dtype=torch.long).to(self.device)
            target_lengths = torch.full(size=(targets.size(0),),
                                        fill_value=cfg.target_size, dtype=torch.long).to(self.device)

        self.optimizer.zero_grad()
        logits = self.model(sources)
        loss = self.loss(logits, targets, source_lengths, target_lengths)
        for weight in self.model.get_weights():
            loss += self.weight_decay * torch.sum(torch.square(weight))

        loss.backward()
        self.optimizer.step()

        logits = logits.cpu().detach().permute(1, 0, 2).argmax(dim=-1).numpy()
        logits = remove_duplicate(logits, cfg.target_size)

        # ===char recognition acc===
        # does not increase when the num of chars not reach the condition
        # so theoretically higher than the log value
        try:
            self.train_acc += np.equal(np.array(logits, dtype=np.object), targets.cpu().numpy()).sum()/\
                            (cfg.target_size*cfg.batch_size)
        except ValueError:
            pass
        self.train_loss += loss.data.item()

    def validate(self, sources, targets):
        if self.device:
            sources = torch.tensor(sources, dtype=torch.float).to(self.device)
            targets = torch.tensor(targets, dtype=torch.long).to(self.device)
            source_lengths = torch.full(size=(sources.size(0),),
                                        fill_value=cfg.logit_size, dtype=torch.long).to(self.device)
            target_lengths = torch.full(size=(targets.size(0),),
                                        fill_value=cfg.target_size, dtype=torch.long).to(self.device)

        logits = self.model(sources)
        loss = self.loss(logits, targets, source_lengths, target_lengths)
        for weight in self.model.get_weights():
            loss += self.weight_decay * torch.sum(torch.square(weight))

        logits = logits.cpu().detach().permute(1, 0, 2).argmax(dim=-1).numpy()
        logits = remove_duplicate(logits, cfg.target_size)
        try:
            self.val_acc += np.equal(np.array(logits, dtype=np.object), targets.cpu().numpy()).sum() /\
                            (cfg.target_size*cfg.batch_size)
        except ValueError:
            pass
        self.val_loss += loss.data.item()

    def generate_sample(self, sources, targets, batch):
        """
        Drawing and labeling
        """
        if self.device:
            sources = torch.tensor(sources, dtype=torch.float).to(self.device)
        logits = self.model(sources).cpu().detach().permute(1, 0, 2).argmax(dim=-1).numpy()

        random_idx = np.random.choice(logits.shape[0], 1)

        source = sources[random_idx].squeeze().permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(np.uint8((source + 1) * 127.5))

        logit = logits[random_idx]
        logit = remove_duplicate(logit, cfg.target_size)

        image_mark(image, logit, batch)
