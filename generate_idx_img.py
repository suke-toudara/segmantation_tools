import os
import cv2
import numpy as np
from PIL import Image
from glob import glob 
import matplotlib.pyplot as plt


input_folder_path = "/home/suke/Desktop/segmantation_tools/data/val_label"
output_folder_path = "/home/suke/Desktop/segmantation_tools/data/val_ano"

# 指定したクラス以外の領域のRGB値を黒に変換
# なぜか領域の境界が変なRGB値になっていたので作成
def paint_tool(input_folder_path):
    # 指定のBGR値以外の部分を(0, 0, 0)に変換
    specified_colors = [(0, 0, 128), (0, 128, 0), (0, 128, 128)]  # 指定のBGR値
    replacement_color = (0, 0, 0)  # 変換後のBGR値
    for filename in os.listdir(input_folder_path):
        if filename.endswith(".png"):
            input_path = os.path.join(input_folder_path, filename)
            output_path = os.path.join(output_folder_path, filename)
            image = cv2.imread(input_path)
            # 指定のBGR値以外の部分を変換
            for color in specified_colors:
                mask = np.all(image != color, axis=-1)  # 指定のBGR値以外のピクセルをTrueにするマスク
                image[mask] = replacement_color
            cv2.imwrite(output_path, image)
    print("画像の変換が完了しました。")



# RGB画像からindexカラー画像を作成
def rgb2mask(input_folder_path):
    """
    turn a 3 channel RGB image to 1 channel index image
    """
    color2index = {
        #B G R
        (0,0,0)     : 0 ,  #Balck        #background 
        (255, 0, 0) : 1 ,  #Red          #ripe_tomato      
        (0, 255, 0) : 2 ,  #Right_Green  #Green_Tomato
        (0, 100, 0) : 3 ,  #Green        #main_stem      
        (0, 0, 255) : 4 ,  #Blue         #peduncle     
    }
    
    """
    ---------------------------------------------------
    """
    for filename in os.listdir(input_folder_path):
        if filename.endswith(".png"):
            input_path = os.path.join(input_folder_path, filename)
            image = Image.open(input_path)
            width, height = image.size
            index_image = np.zeros((height, width), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    rgb = image.getpixel((x, y))[:3]  # 画像のピクセル値を取得（最初の3つの要素）
                    index = color2index.get(rgb, 0)  # マッピングに対応するインデックスを取得（対応がない場合は0を使用）
                    index_image[y, x] = index
            index_image = Image.fromarray(index_image)
            index_image.save(input_path)

def rgb2mask(input_folder_path,output_folder_path,input_label_folder_path,outputt_label_folder_path):
    """
    turn a 3 channel RGB image to 1 channel index image
    """
    color2index = {
        #B G R
        (0,0,0)     : 0 ,  #Balck        #background 
        (255, 0, 0) : 1 ,  #Red          #ripe_tomato      
        (0, 255, 0) : 2 ,  #Right_Green  #Green_Tomato
        (0, 100, 0) : 3 ,  #Green        #main_stem      
        (0, 0, 255) : 4 ,  #Blue         #peduncle     
    }
    
    """
    ---------------------------------------------------
    """
    i = 0
    for filename in os.listdir(input_label_folder_path):
        if filename.endswith(".png"):
            input_label_path = os.path.join(input_label_folder_path, filename)
            output_label_path = os.path.join(outputt_label_folder_path, f"{i}.png")

            image = Image.open(input_label_path)
            width, height = image.size
            index_image = np.zeros((height, width), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    rgb = image.getpixel((x, y))[:3]  # 画像のピクセル値を取得（最初の3つの要素）
                    index = color2index.get(rgb, 0)  # マッピングに対応するインデックスを取得（対応がない場合は0を使用）
                    index_image[y, x] = index
            index_image = Image.fromarray(index_image)
            index_image.save(output_label_path)

            #　対応するカメラ画像も保存
            new_file_name = os.path.splitext(filename)[0].replace("_mask","")
            input_path = os.path.join(input_folder_path, new_file_name+".jpg")
            output_path = os.path.join(output_folder_path, f"{i}.png")
            image = Image.open(input_path)
            image.save(output_path)

            i +=1

def img_viewer(): #train_label_list,idx):
    #img_pth = train_label_list[idx]
    img_pth = "C:/Users/Yasukawa Lab/Desktop/segmantation_tools/dataset/train/78.png"
    img = Image.open(img_pth)
    print(img.size)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    img_label_pth = "C:/Users/Yasukawa Lab/Desktop/segmantation_tools/dataset/train_lable/78.png"
    img_label = Image.open(img_label_pth)
    print(img_label.size)
    plt.imshow(img_label)
    plt.axis("off")
    plt.show()

def main():
    train_path = "C:/Users/Yasukawa Lab/Desktop/segmantation_tools/dataset/train"
    train_path_output_path = "C:/Users/Yasukawa Lab/Desktop/segmantation_tools/dataset/train_label"
    train_label_path = "C:/Users/Yasukawa Lab/Desktop/segmantation_tools/dataset/valid"
    train_label_path_output_path = "C:/Users/Yasukawa Lab/Desktop/segmantation_tools/dataset/vaild_lable"
    
    # val_path = "/home/suke/Desktop/segmantation_tools/data/val_label"
    # val_path_output_path = ""
    # train_label_list = sorted(glob(f'{train_path}/*.png'))
    # val_label_list = sorted(glob(f'{val_path}/*.png'))
    #rgb2mask(train_path,train_path_output_path,train_label_path,train_label_path_output_path)
    #rgb2mask(val_path,val_path_output_path)
    img_viewer()

    # path =  "/home/suke/Desktop/segmantation_tools/data/train_label/tomato1.png"#
    # img = cv2.imread(path)
    # img.resize(960,540)
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    
if __name__ == "__main__" :
    main()