#pytorch module
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
import torch.nn.init as init
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import torch.nn.functional as F

# PSP Net module
from u_net_plus import UNet, UNet_Plus
from dataset import Dataset
from loss import CrossEntropyLoss,DiceLoss
# etc module
import numpy as np
from glob import glob
import argparse
import math
import time

# for debug
from PIL import Image
import matplotlib.pyplot as plt
import os



# param
parser = argparse.ArgumentParser()
parser.add_argument('--img_height', type=int, default=512) #img_height #64の倍率じゃないとだめっぽい
parser.add_argument('--img_width', type=int, default=960) #img_width
parser.add_argument('--batch_size', type=int, default=2) #batch_size
parser.add_argument('--epochs', type=int, default=150) #学習回数
parser.add_argument('--lr', type=float, default=1e-4) #学習率
parser.add_argument('--patience', type=int, default=10) # earlystoppingの監視対象回数
parser.add_argument('--devices', default=1) # or 1
parser.add_argument('--accelerator', default="gpu") #cpu or gpu or tpu")
parser.add_argument('--train_img', type=str, default="/home/suke/Desktop/segmantation_tools/4class_Classification/train")
parser.add_argument('--train_label', type=str, default="/home/suke/Desktop/segmantation_tools/4class_Classification/train_label") 
parser.add_argument('--val_img', type=str, default="/home/suke/Desktop/segmantation_tools/4class_Classification/val") 
parser.add_argument('--val_dataset', type=str, default="/home/suke/Desktop/segmantation_tools/4class_Classification/val_label") 
param = parser.parse_args(args=[])

def main() :
    net = UNet(num_classes=4) 
    #net = UNet_Plus(num_classes=4, input_channels=3, deep_supervision=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    torch.backends.cudnn.benchmark = True

    # make_dataset_list
    train_img_list = sorted(glob(f'{param.train_img}/*.png'))
    train_label_list = sorted(glob(f'{param.train_label}/*.png'))
    val_img_list = sorted(glob(f'{param.val_img}/*.png'))
    val_label_list = sorted(glob(f'{param.val_dataset}/*.png'))
    # make DataLoader
    train_dataset = Dataset(train_img_list,train_label_list,"train", param.img_height, param.img_width)
    val_dataset = Dataset(val_img_list, val_label_list,"val", param.img_height, param.img_width)
    train_dataloader = data.DataLoader(train_dataset, batch_size=param.batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=param.batch_size, shuffle=False)
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    # training param
    #TODO clasweight 実装したい
    # backgrownd = 0.0001
    # accessory_stem = 0.5
    # main_stem = 0.5
    # class_weights = torch.tensor([backgrownd, accessory_stem, main_stem])
    
    criterion = CrossEntropyLoss().to(device)#nn.BCEWithLogitsLoss().to(device) #nn.BCELoss()#DiceLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.8)
    scheduler = LambdaLR(optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)


    train_losses = []
    print("strat train")
    for epoch in range(param.epochs):
        print('-------------')
        print(f'Epoch {epoch+1}/{param.epochs}')
        print('-------------')        
        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss = 0.0  
        epoch_val_loss = 0.0  

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train() 
                for imges, anno_class_imges in dataloaders_dict[phase]:
                    optimizer.zero_grad() #init_grad
                    device = "cuda"
                    imges = imges.to(device)
                    outputs = net(imges)                   
                    anno_class_imges = anno_class_imges.to(device)
                    loss = criterion(outputs, anno_class_imges.long())   
                    loss.backward()
                    optimizer.step()
                    scheduler.step()  
                print('train')

            else:
                if((epoch+1) % 5 == 0):
                    net.eval()   
                    print('-------------')
                    print('val')
                else:
                    continue
        
        t_epoch_finish = time.time()
        print('-------------')
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        if((epoch+1) % 50 == 0):
            torch.save(net.state_dict(), '/home/suke/Desktop/segmantation_tools/u_net++/weights' +str(epoch+1) + '.pth')
    # save modul
    torch.save(net.state_dict(), '/home/suke/Desktop/segmantation_tools/u_net++/weights' +str(epoch+1) + '.pth')


if __name__ == "__main__" :
   main()