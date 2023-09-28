
import sys
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
import argparse

import PIL
from PIL import Image

# pytorch_lightning modul
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torchmetrics
from torchmetrics.functional import accuracy,jaccard_index

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torchvision.models.segmentation import deeplabv3
from  dataset  import Dataset
import os

from deeplab3 import Deeplabv3

# ハイパーパラメータの設定
parser = argparse.ArgumentParser()
parser.add_argument('--image_width', type=int, default=475) #image_size
parser.add_argument('--image_height', type=int, default=475) #image_size
parser.add_argument('--batch_size', type=int, default=2) #batch_size
parser.add_argument('--epochs', type=int, default=20) #学習回数
parser.add_argument('--lr', type=float, default=1e-4) #学習率
parser.add_argument('--patience', type=int, default=10) # earlystoppingの監視対象回数
parser.add_argument('--devices', default=1) # or 1
parser.add_argument('--accelerator', default="gpu") #cpu or gpu or tpu")
parser.add_argument('--train_img', type=str, default="/home/suke/Desktop/segmantation_tools/4class_Classification/train")
parser.add_argument('--train_label', type=str, default="/home/suke/Desktop/segmantation_tools/4class_Classification/train_label") 
parser.add_argument('--val_img', type=str, default="/home/suke/Desktop/segmantation_tools/4class_Classification/val") 
parser.add_argument('--val_dataset', type=str, default="/home/suke/Desktop/segmantation_tools/4class_Classification/val_label") 
param = parser.parse_args(args=[])

SAVE_MODEL_PATH = f'./weights'  #save_model_path
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

# make_dataset_list
train_img_list = sorted(glob(f'{param.train_img}/*.png'))
train_label_list = sorted(glob(f'{param.train_label}/*.png'))
val_img_list = sorted(glob(f'{param.val_img}/*.png'))
val_label_list = sorted(glob(f'{param.val_dataset}/*.png'))
# make DataLoader
train_dataset = Dataset(train_img_list,train_label_list,"train", param.image_width,param.image_height)
val_dataset = Dataset(val_img_list, val_label_list,"val", param.image_width,param.image_height)
train_dataloader = data.DataLoader(train_dataset, batch_size=param.batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=param.batch_size, shuffle=False)
dataloader = {"train": train_dataloader, "val": val_dataloader}




model_checkpoint = ModelCheckpoint(
    dirpath =SAVE_MODEL_PATH,           #directory to save the model file.
    filename="DeepLabV3_resnet101"+"{epoch:02d}-{val_loss:.2f}", #checkpoint filename
    monitor='val_loss',                 #モニターする数量
    verbose = False ,                   #verbosity mode
    save_top_k=1,                       #精度の良いモデルのうち上からn個データを保存する
    mode='min',                         #min or max データを保存するときにmonitorで設定した値がmin or maxのモデルを保存する
    auto_insert_metric_name = False,    #Trueの場合、チェックポイントのファイル名にはメトリック名が含まれます。
    save_weights_only = False,          #Trueの場合は重みだけ保存される.optimizer,lrなどの学習状況は保存されない 
    #every_n_train_steps = None,        #every_n_train_steps == None または every_n_train_steps == 0 の場合、
                                        #トレーニング中の保存をスキップします。無効にするには、every_n_train_steps = 0を設定します。
                                        #この値はNoneまたは非負でなければなりません。これはtrain_time_intervalおよびevery_n_epochsと互いに排他的でなければならない。
    #train_time_interval = ,            #チェックポイントは指定された時間間隔で監視されます。
    #every_n_epochs = ,                 # チェックポイント間のエポック数
    #save_on_train_epoch_end = ,        #トレーニング エポックの終了時にチェックポイントを実行するかどうか。これが の場合False、検証の最後にチェックが実行されます。
)

early_stopping = EarlyStopping(
    monitor='val_loss', #打ち切るために監視する値
    patience=param.patience, # 改善が見られないチェックの回数を超えると、トレーニングが停止される。
    #verbose = False , #verbosity mode.
    mode='min', #最小のときはカウントしない
    #strict = False ,#Trueのとき 参照する値がモニターで見つからない場合にトレーニングをクラッシュさせる
    #check_finite = False , # Trueのとき、参照するあたいが NaN または無限になったときにトレーニングを停止
    #stopping_threshold = 0.001 , #監視対象の量がこのしきい値に達したら、直ちにトレーニングを停止
    #divergence_threshold = 100 , #監視された量がこのしきい値よりも悪くなったらすぐにトレーニングを停止
    #check_on_train_epoch_end = False , #whether to run early stopping at the end of the training epoch. 
                                        #If this is False, then the check runs at the end of the validation.
    #log_rank_zero_only( bool)          #Trueの場合、logs the status of the early stopping callback only for rank 0
)

#pl.seed_everything(0) #Lightningのシード（乱数の種）を設定する

# train phase
trainer = pl.Trainer(
    devices=param.devices,
    accelerator=param.accelerator,
    # strategy="ddp" ,
    default_root_dir=SAVE_MODEL_PATH,
    max_epochs=param.epochs, 
    callbacks=[model_checkpoint,early_stopping], #add checkpoint and early_stopping
    profiler="simple",
    )

model = Deeplabv3(num_class=4) 

trainer.fit(model=model, train_dataloaders=dataloader['train'], val_dataloaders=dataloader['val']) #引数はモデル,学習用データローダ,評価用データローダ
