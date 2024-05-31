# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

import argparse
from pathlib import Path
import os
import tarfile
import shutil
import json

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

DEFAULT_DATA_DIR = './data'
DOWNLOAD_DIR = './data/download'
TRAIN_IMAGES_URL = 'http://images.cocodataset.org/zips/train2017.zip'
VALID_IMAGES_URL = 'http://images.cocodataset.org/zips/val2017.zip'
ANNOTATION_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
CLASS80_NAME_TO_LABEL = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13, 'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34, 'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 'bottle': 39, 'wine glass': 40, 'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48, 'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 'chair': 56, 'couch': 57, 'potted plant': 58, 'bed': 59, 'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67, 'microwave': 68, 'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 'vase': 75, 'scissors': 76, 'teddy bear': 77, 'hair drier': 78, 'toothbrush': 79}
CLASS80_LABEL_TO_NAME = {val: key for key, val in CLASS80_NAME_TO_LABEL.items()}


def txtywh2cxcywh(top_left_x, top_left_y, width, height):
    cx = top_left_x + (width / 2)
    cy = top_left_y + (height / 2)
    w = width
    h = height
    return cx, cy, w, h


def cxcywh2cxcywhn(cx, cy, w, h, img_w, img_h):
    cx = cx / img_w
    cy = cy / img_h
    w = w / img_w
    h = h / img_h
    return cx, cy, w, h


if __name__ == '__main__':
    # Set argument (data directory)
    parser = argparse.ArgumentParser(description="Parser for coco2017 dataset downloader.")
    parser.add_argument('--dir', type=str, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    # Download coco2017 dataset
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    train_images_download_path = Path(DOWNLOAD_DIR) / 'train2017.zip'
    if train_images_download_path.exists():
        print(f'Download path {train_images_download_path} already exists! download step is skipped.')
    else:
        torch.hub.download_url_to_file(TRAIN_IMAGES_URL, train_images_download_path)

    valid_images_download_path = Path(DOWNLOAD_DIR) / 'val2017.zip'
    if valid_images_download_path.exists():
        print(f'Download path {valid_images_download_path} already exists! download step is skipped.')
    else:
        torch.hub.download_url_to_file(VALID_IMAGES_URL, valid_images_download_path)

    ann_download_path = Path(DOWNLOAD_DIR) / 'annotaions_trainval2017.zip'
    if ann_download_path.exists():
        print(f'Download path {ann_download_path} already exists! download step is skipped.')
    else:
        torch.hub.download_url_to_file(ANNOTATION_URL, ann_download_path)

    coco2017_path = Path(args.dir) / 'coco2017'
    os.makedirs(coco2017_path, exist_ok=True)

    # Unzip train images
    print('Unzip training images zip file ...')
    images_dir = coco2017_path / 'images'
    shutil.unpack_archive(train_images_download_path, images_dir, "zip")
    print('Rename train2017 to train')
    try: # Remove already exists one
        shutil.rmtree(images_dir / 'train')
    except OSError as e:
        print(e)
    os.rename(images_dir / 'train2017', images_dir / 'train')
    print('Done!')

    # Unzip valid images
    print('Unzip training images zip file ...')
    shutil.unpack_archive(valid_images_download_path, images_dir, "zip")
    print('Rename val2017 to valid')
    try: # Remove already exists one
        shutil.rmtree(images_dir / 'valid')
    except OSError as e:
        print(e)
    os.rename(images_dir / 'val2017', images_dir / 'valid')
    print('Done!')

    # Unzip annotation zip file
    print('Unzip annotation zip file ...')
    shutil.unpack_archive(ann_download_path, coco2017_path, "zip")
    print('Done!')

    # Reformat train annotaion to yolo format
    # TODO: Support yolo format and leave yolo format as optional
    print('Building train labels ...')
    train_annotation_path = coco2017_path / 'annotations' / 'instances_train2017.json'
    train_label_dir = coco2017_path / 'labels' / 'train'
    os.makedirs(train_label_dir, exist_ok=True)

    with open(train_annotation_path) as f:
        train_ann_json = json.load(f)

    category = train_ann_json['categories']
    CLASS91_LABEL_TO_NAME = {}
    for item in category:
        CLASS91_LABEL_TO_NAME[item['id']] = item['name']

    train_annotations = {image_info['id']: [image_info['file_name']] for image_info in train_ann_json['images']}
    train_imgid_to_info = {info['id']: info for info in train_ann_json['images']}
    for ann in tqdm(train_ann_json['annotations']):
        image_id = ann['image_id']
        
        category_id = ann['category_id']
        label = CLASS80_NAME_TO_LABEL[CLASS91_LABEL_TO_NAME[category_id]]
        
        # TODO: Support various box type e.g. xyxy
        top_left_x, top_left_y, width, height  = ann['bbox']
        cx, cy, w, h = txtywh2cxcywh(top_left_x, top_left_y, width, height)
        cx, cy, w, h = cxcywh2cxcywhn(cx, cy, w, h, train_imgid_to_info[image_id]['width'], train_imgid_to_info[image_id]['height'])

        instance = [label, cx, cy, w, h]
        train_annotations[image_id].append(instance)

    for image_id, info in tqdm(train_annotations.items()):
        file_name = info[0]
        
        texts = ''
        if len(info) != 1:
            for line in info[1:]:
                texts += f'{line[0]} {line[1]} {line[2]} {line[3]} {line[4]}\n'

        with open((train_label_dir / file_name).with_suffix('.txt'), 'w') as f:
            f.write(texts)

    # Reformat valid annotaion to yolo format
    print('Building valid labels ...')
    valid_annotation_path = coco2017_path / 'annotations' / 'instances_val2017.json'
    valid_label_dir = coco2017_path / 'labels' / 'valid'
    os.makedirs(valid_label_dir, exist_ok=True)

    with open(valid_annotation_path) as f:
        valid_ann_json = json.load(f)

    valid_annotations = {image_info['id']: [image_info['file_name']] for image_info in valid_ann_json['images']}
    valid_imgid_to_info = {info['id']: info for info in valid_ann_json['images']}
    for ann in tqdm(valid_ann_json['annotations']):
        image_id = ann['image_id']
        
        category_id = ann['category_id']
        label = CLASS80_NAME_TO_LABEL[CLASS91_LABEL_TO_NAME[category_id]]
        
        # TODO: Support various box type e.g. xyxy
        top_left_x, top_left_y, width, height  = ann['bbox']
        cx, cy, w, h = txtywh2cxcywh(top_left_x, top_left_y, width, height)
        cx, cy, w, h = cxcywh2cxcywhn(cx, cy, w, h, valid_imgid_to_info[image_id]['width'], valid_imgid_to_info[image_id]['height'])

        instance = [label, cx, cy, w, h]
        valid_annotations[image_id].append(instance)
    
    for image_id, info in tqdm(valid_annotations.items()):
        file_name = info[0]
        
        texts = ''
        if len(info) != 1:
            for line in info[1:]:
                texts += f'{line[0]} {line[1]} {line[2]} {line[3]} {line[4]}\n'

        with open((valid_label_dir / file_name).with_suffix('.txt'), 'w') as f:
            f.write(texts)

    # Build id_mapping
    id_mapping = [CLASS80_LABEL_TO_NAME[i] for i in range(80)]
    with open(coco2017_path / 'id_mapping.json', 'w') as f:
        json.dump(id_mapping, f)

    try:
        shutil.rmtree(coco2017_path / 'annotations')
    except OSError as e:
        print(e)