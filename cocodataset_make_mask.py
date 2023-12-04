# import json
# from PIL import Image, ImageDraw
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import os
# import numpy as np

# # COCOアノテーションJSONファイルのパス
# json_path = "C:/Users/Yasukawa Lab/Desktop/segmantation_tools/hibikino_segmentation.v1i.coco-segmentation/train/_annotations.coco.json"

# with open(json_path, "r") as f:
#     coco_data = json.load(f)

# # マスク画像保存用のディレクトリ
# output_dir = "C:/Users/Yasukawa Lab/Desktop/segmantation_tools/hibikino_segmentation.v1i.coco-segmentation/outputs"

# pre_id = 0

# color_map = {
#                 "ripe_tomato" : {},
#                 "green_tomato" : {},
#                 "main_stem" : {},
#                 "peduncle" : {},

#             }
# # 各画像ごとに処理を行う
# for annotation in coco_data["annotations"]:
#     # 画像ID
#     image_id = annotation["image_id"]
#     print(image_id)

#     # 画像ファイル名
#     image_info = next(item for item in coco_data["images"] if item["id"] == image_id)
#     image_file_name = image_info["file_name"]
#     # print(image_file_name)

#     width = image_info["width"]
#     height = image_info["height"]


#     # マスク画像の保存パス
#     mask_path = os.path.join(output_dir, image_file_name.replace(".jpg", "_mask.png"))
    
    
#     # マスク画像が存在する場合は読み込み、ない場合は新規作成
#     if os.path.exists(mask_path):
#         mask = Image.open(mask_path).convert("L")
#     else:
#         mask = Image.new("L", (width, height), 0)

#     draw = ImageDraw.Draw(mask)

#     # セグメンテーション情報からポリゴン描画
#     for segmentation in annotation["segmentation"]:
#         draw.polygon(segmentation, outline=1, fill=1)

    

#     # マスク画像の保存
#     mask.save(mask_path)

#     # マスク画像を描画
#     plt.figure()
#     plt.imshow(mask, cmap="gray")
#     plt.title("Mask for {}".format(image_file_name))
#     plt.show()

#     pre_id = image_id 


import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

# COCOアノテーションJSONファイルのパス
json_path = "C:/Users/Yasukawa Lab/Desktop/segmantation_tools/hibikino_segmentation.v1i.coco-segmentation/train/_annotations.coco.json"

with open(json_path, "r") as f:
    coco_data = json.load(f)

# マスク画像保存用のディレクトリ
output_dir = "C:/Users/Yasukawa Lab/Desktop/segmantation_tools/hibikino_segmentation.v1i.coco-segmentation/outputs"

# 各カテゴリごとに異なる色を割り当てる
# color_map = {
#     "ripe_tomato": (255, 0, 0),  # Red
#     "green_tomato": (0, 255, 0),  # Green
#     "main_stem": (0, 0, 255),  # Blue
#     "peduncle": (255, 255, 0),  # Yellow
# }

color_map = {
    1 : (0, 255, 0),  #Green_Tomato #Right_Green
    2 : (0, 100, 0),  #main_stem    #Green  
    3 : (0, 0, 255),  #peduncle     #Blue
    4 : (255, 0, 0),  #ripe_tomato  #Red 
}


# 各画像ごとに処理を行う
for annotation in coco_data["annotations"]:
    # 画像ID
    image_id = annotation["image_id"]


    # 画像ファイル名
    image_info = next(item for item in coco_data["images"] if item["id"] == image_id)
    image_file_name = image_info["file_name"]
    # print(image_file_name)

    width = image_info["width"]
    height = image_info["height"]

    # マスク画像の保存パス
    mask_path = os.path.join(output_dir, image_file_name.replace(".jpg", "_mask.png"))

    # マスク画像が存在する場合は読み込み、ない場合は新規作成
    if os.path.exists(mask_path):
        mask = Image.open(mask_path).convert("RGB")
    else:
        mask = Image.new("RGB", (width, height), (0, 0, 0))

    draw = ImageDraw.Draw(mask)

    # セグメンテーション情報からポリゴン描画
    for segmentation in annotation["segmentation"]:
        category = annotation["category_id"]
        color = color_map.get(category, (0, 0, 0))  # マッピングがない場合は黒色を使用
        draw.polygon(segmentation, outline=color, fill=color)

    # マスク画像の保存
    mask.save(mask_path)

    # マスク画像を描画
    # plt.figure()
    # plt.imshow(mask)
    # plt.title("Mask for {}".format(image_file_name))
    # plt.show()