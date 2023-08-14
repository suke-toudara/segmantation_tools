import matplotlib.pyplot as plt

from glob import glob
import argparse

import PIL
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torchmetrics
from torchmetrics.functional import accuracy, iou

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.segmentation import deeplabv3

#pytorch_lightningを継承しているのでそれに沿った書き方をする
"""
__init__(*args, **kwargs)	: modelやcriterionなどをクラス変数として設定する。
forward(*args, **kwargs) : nn.Moduleのforwardと同じだが、主に予測で使用する。クラス内ではself(batch)として呼び出される。
training_step(batch, batch_idx, optimizer_idx, hiddens) : DataLaoderをイテレーションして出力したbatchを引数として受け取り、criterionで計算したlossをreturnする。forwardとは独立したメソッド。
validation_step(batch, batch_idx, dataloader_idx) : DataLaoderをイテレーションして出力したbatchを引数として受け取り、メトリックを計算する。
configure_optimizers() : optimizerをreturnする。schedulerを使用する場合はreturnをoptimizerのリストとschedulerのリストのタプルとする。
"""

class Deeplabv3(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(weights='DEFAULT') #'COCO_WITH_VOC_LABELS_V1')
        self.model.classifier = deeplabv3.DeepLabHead(2048, 3)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        h = self.model(x)
        return h

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)['out']
        target = x.squeeze(0)
        loss = self.loss(out.squeeze(0), target)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch #batch : train val  test(設定している場合は)
        out = self.model(x)['out']
        target = x.squeeze(0)
        loss = self.loss(out.squeeze(0), target)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def backward(self, loss):
        loss.backward()

    #define the optimizer(s) for your models.
    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer