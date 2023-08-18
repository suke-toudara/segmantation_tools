import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import numpy as np

class Compose(object):
    """引数transformに格納された変形を順番に実行するクラス
       対象画像とアノテーション画像を同時に変換させます。 
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, anno_class_img):
        for t in self.transforms:
            img, anno_class_img = t(img, anno_class_img)
        return img, anno_class_img


# class Scale(object):
#     def __init__(self, scale):
#         self.scale = scale

#     def __call__(self, img, anno_class_img):

#         width = img.size[0]  # img.size=[幅][高さ]
#         height = img.size[1]  # img.size=[幅][高さ]

#         # 拡大倍率をランダムに設定
#         scale = np.random.uniform(self.scale[0], self.scale[1])

#         scaled_w = int(width * scale)  # img.size=[幅][高さ]
#         scaled_h = int(height * scale)  # img.size=[幅][高さ]

#         # 画像のリサイズ
#         img = img.resize((scaled_w, scaled_h), Image.BICUBIC)

#         # アノテーションのリサイズ
#         anno_class_img = anno_class_img.resize(
#             (scaled_w, scaled_h), Image.NEAREST)

#         # 画像を元の大きさに
#         # 切り出し位置を求める
#         if scale > 1.0:
#             left = scaled_w - width
#             left = int(np.random.uniform(0, left))

#             top = scaled_h-height
#             top = int(np.random.uniform(0, top))

#             img = img.crop((left, top, left+width, top+height))
#             anno_class_img = anno_class_img.crop(
#                 (left, top, left+width, top+height))

#         else:
#             # input_sizeよりも短い辺はpaddingする
#             p_palette = anno_class_img.copy().getpalette()

#             img_original = img.copy()
#             anno_class_img_original = anno_class_img.copy()

#             pad_width = width-scaled_w
#             pad_width_left = int(np.random.uniform(0, pad_width))

#             pad_height = height-scaled_h
#             pad_height_top = int(np.random.uniform(0, pad_height))

#             img = Image.new(img.mode, (width, height), (0, 0, 0))
#             img.paste(img_original, (pad_width_left, pad_height_top))

#             anno_class_img = Image.new(
#                 anno_class_img.mode, (width, height), (0))
#             anno_class_img.paste(anno_class_img_original,
#                                  (pad_width_left, pad_height_top))
#             anno_class_img.putpalette(p_palette)

#         return img, anno_class_img


class RandomRotation(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, anno_class_img):

        # 回転角度を決める
        rotate_angle = (np.random.uniform(self.angle[0], self.angle[1]))

        # 回転
        img = img.rotate(rotate_angle, Image.BILINEAR)
        anno_class_img = anno_class_img.rotate(rotate_angle, Image.NEAREST)

        return img, anno_class_img


class RandomMirror(object):
    """50%の確率で左右反転させるクラス"""

    def __call__(self, img, anno_class_img):
        if np.random.randint(2):
            img = ImageOps.mirror(img)
            anno_class_img = ImageOps.mirror(anno_class_img)
        return img, anno_class_img


class Resize(object):
    """引数input_sizeに大きさを変形するクラス"""

    def __init__(self, image_width,image_height):
        self.image_width = image_width
        self.image_height = image_height

    def __call__(self, img, anno_class_img):
        img = img.resize((self.image_height,self.image_width),
                         Image.BICUBIC)
        anno_class_img = anno_class_img.resize(
            (self.image_height,self.image_width), Image.NEAREST)

        return img, anno_class_img


class Normalize_Tensor(object):
    def __init__(self, color_mean, color_std):
        self.color_mean = color_mean
        self.color_std = color_std

    def __call__(self, img, anno_class_img):

        # PIL画像をTensorに。大きさは最大1に規格化される
        img = transforms.functional.to_tensor(img)

        # 色情報の標準化
        img = transforms.functional.normalize(
            img, self.color_mean, self.color_std)

        # アノテーション画像をNumpyに変換
        anno_class_img = np.array(anno_class_img)  # [高さ][幅]

        # 'ambigious'には255が格納されているので、0の背景にしておく
        index = np.where(anno_class_img == 255)
        anno_class_img[index] = 0

        # アノテーション画像をTensorに
        anno_class_img = torch.from_numpy(anno_class_img)

        return img, anno_class_img



class DataTransform():
    """
    画像とアノテーションの前処理クラス。訓練時と検証時で異なる動作をする。
    画像のサイズをinput_size x input_sizeにする。
    訓練時はデータオーギュメンテーションする。


    Attributes
    ----------
    input_size : int
        リサイズ先の画像の大きさ。
    color_mean : (R, G, B)
        各色チャネルの平均値。
    color_std : (R, G, B)
        各色チャネルの標準偏差。
    """

    def __init__(self, image_width,image_height, color_mean, color_std):
        self.data_transform = {
            'train': Compose([
                #Scale(scale=[0.5, 1.5]),  # 画像の拡大
                RandomRotation(angle=[-10, 10]),  # 回転
                RandomMirror(),  # ランダムミラー
                Resize(image_width,image_height),  # リサイズ(input_size)
                Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ]),
            'val': Compose([
                Resize(image_width,image_width),  # リサイズ(input_size)
                Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img, anno_class_img)