import PIL
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
from data_augumentation import DataTransform

"""
--- input ---
img_path_list : 元画像のpath
label_path_list : ラベルデータのpath
image_size : 入力サイズ数

--- output ---
img,label: Data Augmentationを行った画像とラベル
"""
class Dataset(data.Dataset):
    def __init__(self, img_path_list, label_path_list, phase, image_width,image_height):
        self.image_path_list = img_path_list
        self.label_path_list = label_path_list
        self.phase = phase
        color_mean = (0.485, 0.456, 0.406)
        color_std = (0.229, 0.224, 0.225)
        self.transform = DataTransform(image_width=image_width, image_height = image_height , color_mean=color_mean, color_std=color_std)
        
    def __len__(self):
        return len(self.image_path_list)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_path_list[idx])
        label = Image.open(self.label_path_list[idx])
        img , label = self.transform(self.phase, img, label)
        return img, label
     