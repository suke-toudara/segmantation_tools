#pytorch module
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
import torch.nn.init as init
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

# PSP Net module
from loss import AuxLoss
from psp_net import PSPNet
from dataset import Dataset

# etc module
import numpy as np
from glob import glob
import argparse
import math
import time

# for debug
from PIL import Image
import matplotlib.pyplot as plt
from data_augumentation import DataTransform
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def debug():
    print("debug")
    # idx = 0
    # img = Image.open(train_img_list[idx])
    # anno_class_img = Image.open(train_label_list[idx])

    # # transform = transforms.Compose(
    # #     [transforms.Resize(475), transforms.CenterCrop(224), transforms.ToTensor()]
    # # )
    # phase = "val"
    # img, anno_class_img = transform(phase, img, anno_class_img)

    # x = img.unsqueeze(0)  # ミニバッチ化：torch.Size([1, 3, 475, 475])
    # outputs = net(x)
    # y = outputs[0]  # AuxLoss側は無視 yのサイズはtorch.Size([1, 21, 475, 475])
    # y = y[0].detach().numpy()  # y：torch.Size([1, 21, 475, 475])
    # y = np.argmax(y, axis=0)
    # plt.imshow(y)
    # plt.show()

    

# ハイパーパラメータの設定
parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=256) #image_size
parser.add_argument('--batch_size', type=int, default=4) #batch_size
parser.add_argument('--epochs', type=int, default=1) #学習回数
parser.add_argument('--lr', type=float, default=1e-4) #学習率
parser.add_argument('--patience', type=int, default=10) # earlystoppingの監視対象回数
parser.add_argument('--devices', default=1) # or 1
parser.add_argument('--accelerator', default="gpu") #cpu or gpu or tpu")
parser.add_argument('--model_name', type=str, default="deeplabv3") #model_name  
parser.add_argument('--train_img', type=str, default="/home/suke/Desktop/segmantation_tools/data/train")
parser.add_argument('--train_label', type=str, default="/home/suke/Desktop/segmantation_tools/data/train_label") 
parser.add_argument('--val_img', type=str, default="/home/suke/Desktop/segmantation_tools/data/val") 
parser.add_argument('--val_dataset', type=str, default="/home/suke/Desktop/segmantation_tools/data/val_label") 
param = parser.parse_args(args=[])

def main() :
    #
    if param.model_name == PSPNet :
        net=PSPNet(n_classes=4)
    else :
        net=PSPNet(n_classes=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    torch.backends.cudnn.benchmark = True

    # make_dataset_list
    train_img_list = sorted(glob(f'{param.train_img}/*.png'))
    train_label_list = sorted(glob(f'{param.train_label}/*.png'))
    val_img_list = sorted(glob(f'{param.val_img}/*.png'))
    val_label_list = sorted(glob(f'{param.val_dataset}/*.png'))
    # make DataLoader
    train_dataset = Dataset(train_img_list,train_label_list,"train", param.image_size)
    val_dataset = Dataset(val_img_list, val_label_list,"val", param.image_size)
    train_dataloader = data.DataLoader(train_dataset, batch_size=param.batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=param.batch_size, shuffle=False)
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    # training param
    #TODO clasweight 実装したい
    # backgrownd = 0.0001
    # accessory_stem = 0.5
    # main_stem = 0.5
    # class_weights = torch.tensor([backgrownd, accessory_stem, main_stem])
    # criterion = AuxLoss(aux_weight=0.4, class_weights=class_weights)
    criterion = AuxLoss(aux_weight=0.4)
    
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = LambdaLR(optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)
    batch_multiplier = 3

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
                    imges = imges.to(device)
                    outputs = net(imges)
                    anno_class_imges = anno_class_imges.to(device)
                    loss = criterion(outputs, anno_class_imges.long()) / batch_multiplier   
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

        
        # epochのphaseごとのlossと正解率
        t_epoch_finish = time.time()
        print('-------------')
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

    # 最後のネットワークを保存する
    torch.save(net.state_dict(), '/home/suke/Desktop/segmantation_tools/weight/pspnet' +str(epoch+1) + '.pth')


if __name__ == "__main__" :
    main()