# coding=UTF-8
import os
from csv2coco import Csv2CoCo
import pandas as pd
import numpy as np
import random
import cv2
import json
split_radio = 0.7
work_dir = "/home/test/wx/" # 工作目录
tsv_path = "/data/dataset/test/annotations/" # tsv文件夹
jpg_path = "/data/dataset/test/JPG/" # 所有node.js转过的图都在此文件夹下
csv_path = "/data/dataset/test/coco/coco.csv"
csv_add_tag_path = "/data/dataset/test/coco/coco_add_tag.csv"
# image_dir = "/data/dataset/test/ice-coco/" #重命名后的图像目录
coco_path = "/data/dataset/test/coco"      # coco工作目录
###############################################
#所有的tsv转成一个大csv,所有的标记标签均为sign
###############################################
path_list = os.listdir(tsv_path)
lines = []
for file_name in path_list:
    file = os.path.join(tsv_path,file_name)
    file_name = file_name.split(".")[0]
    with open(file,"r") as f:
        g = f.readlines()
        for line in g:
            line = line.split("\t")
            line = line[:5]
            line = [c[1:] for c in line]
            line[0] = line[0].split(".")[0] + ".jpg"
            line[0] = file_name +"_"+ line[0]
            line = ",".join(line)+",sign"  # 默认所有的标记均为sign
            lines.append(line)
lines = "\n".join(lines)
with open(csv_path,"w") as f:
    f.writelines(lines)
###############################################
#所有的tsv转成一个大csv,所有的标记标签均为x.x.0序号类
###############################################
path_list = os.listdir(tsv_path)
lines = []
for file_name in path_list:
    file = os.path.join(tsv_path,file_name)
    file_name = file_name.split(".")[0]
    with open(file,"r") as f:
        g = f.readlines()
        for line in g:
            line = line.split("\t")
            line = line[:6]
            line = [c[1:] for c in line]
            line[0] = line[0].split(".")[0]
            line[0] = file_name +"_"+ line[0]
            line = ",".join(line)+",sign"  # 默认所有的标记均为sign
            lines.append(line)
lines = "\n".join(lines)
with open(csv_add_tag_path,"w") as f:
    f.writelines(lines)


####################################
#csv 转 json
####################################
total_csv_annotations = {}
annotations = pd.read_csv(csv_path,header=None).values
for annotation in annotations:
    #key = annotation[0].split(os.sep)[-1]
    key = annotation[0]
    #print(key)
    value = np.array([annotation[1:]])
    if key in total_csv_annotations.keys():
        total_csv_annotations[key] = np.concatenate((total_csv_annotations[key],value),axis=0)
    else:
        total_csv_annotations[key] = value
total_keys = list(total_csv_annotations.keys())
random.shuffle(total_keys)

train_keys, val_keys = total_keys[:int(len(total_keys)*split_radio)],total_keys[int(len(total_keys)*split_radio):]
print("train_n:", len(train_keys), 'val_n:', len(val_keys))
l2c_train = Csv2CoCo(image_dir="",total_annos=total_csv_annotations)
train_instance = l2c_train.to_coco(train_keys)
l2c_train.save_coco_json(train_instance, '%s/annotations/instances_train2017.json'%coco_path)

l2c_val = Csv2CoCo(image_dir="",total_annos=total_csv_annotations)
val_instance = l2c_val.to_coco(val_keys)
l2c_val.save_coco_json(val_instance, '%s/annotations/instances_val2017.json'%coco_path)


############################
#将图片提取出来到coco目录
############################
# train
# sample: 2018-02-07_1349_right_000990.jpg
#          /data/dataset/test/JPG/2018-03-23_1358_right/
for key in train_keys:
    src_image_name = key.split("_")[-1]
    dst_image_path = os.path.join(os.path.join(coco_path, "train2017"),src_image_name)
    folder = "_".join(key.split("_")[:3])
    folder = os.path.join(jpg_path,folder)
    src_image = os.path.join(jpg_path,src_image_name)
    image = cv2.imread(src_image)
    cv2.imwrite(dst_image_path,image)
# test
for key in val_keys:
    src_image_name = key.split("_")[-1]
    dst_image_path = os.path.join(os.path.join(coco_path, "val2017"),src_image_name)
    folder = "_".join(key.split("_")[:3])
    folder = os.path.join(jpg_path,folder)
    src_image = os.path.join(jpg_path,src_image_name)
    image = cv2.imread(src_image)
    cv2.imwrite(dst_image_path,image)
data = {"train":train_keys,"val":val_keys}
with open("train_list.json", "w", encoding="UTF-8") as f:
    s = json.dump(data, f, ensure_ascii=False)


