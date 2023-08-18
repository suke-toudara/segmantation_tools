import PIL
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import torch 
"""
--- 入力 ---
img_path_list : 元画像のpath
label_path_list : ラベルデータのpath
image_size : 入力サイズ数

--- 出力 ---
img,label: Data Augmentationを行った画像とラベル
"""
class Dataset(data.Dataset):
    def __init__(self, img_path_list, label_path_list, phase, img_height, img_width):
        self.image_path_list = img_path_list
        self.label_path_list = label_path_list
        self.img_height = img_height
        self.img_width = img_width
        self.phase = phase
        if phase == 'train' :
            self.transform = transforms.Compose([
                    transforms.Resize((img_height, img_width)),
                    #transforms.CenterCrop(224),
                    #Pad(padding, fill=0, padding_mode='constant')
                    #transform = transforms.GaussianBlur(kernel_size=3)
                    transforms.ToTensor(),
                ])
        else :
            self.transform = transforms.Compose([
                    transforms.Resize((img_height, img_width)),
                    transforms.ToTensor(),
                ])

    def __len__(self):
        return len(self.image_path_list)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_path_list[idx])
        label = Image.open(self.label_path_list[idx])
        img = self.transform(img)

        #index画像は普通にto tensor にすると何故かindexの一次元たされるのでtorch.from_numpyでtensor化
        # to teonsor ⇨　[2,1,450,450] torch.from_numpy ⇨　[2,450,450]
        label = label.resize((self.img_width,self.img_height), Image.NEAREST)
        label = np.array(label) 
        label = torch.from_numpy(label)
        return img, label
     