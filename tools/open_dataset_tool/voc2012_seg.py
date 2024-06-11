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
from PIL import Image

DEFAULT_DATA_DIR = './data'
DOWNLOAD_DIR = './data/download'
VOC2012_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
VOC2012_ID_MAPPING_RGB = {
    '(0, 0, 0)': 'background',
    '(128, 0, 0)': 'aeroplane',
    '(0, 128, 0)': 'bicycle',
    '(128, 128, 0)': 'bird',
    '(0, 0, 128)': 'boat',
    '(128, 0, 128)': 'bottle',
    '(0, 128, 128)': 'bus',
    '(128, 128, 128)': 'car',
    '(64, 0, 0)': 'cat',
    '(192, 0, 0)': 'chair',
    '(64, 128, 0)': 'cow',
    '(192, 128, 0)': 'diningtable',
    '(64, 0, 128)': 'dog',
    '(192, 0, 128)': 'horse',
    '(64, 128, 128)': 'motorbike',
    '(192, 128, 128)': 'person',
    '(0, 64, 0)': 'pottedplant',
    '(128, 64, 0)': 'sheep',
    '(0, 192, 0)': 'sofa',
    '(128, 192, 0)': 'train',
    '(0, 64, 128)': 'tvmonitor',
}


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':
    # Set argument (data directory)
    parser = argparse.ArgumentParser(description="Parser for VOC 2012 dataset downloader.")
    parser.add_argument('--dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--label_image_mode', type=str, default='L', choices=['RGB', 'L'], help='Label image mode (RGB or L)')
    args = parser.parse_args()

    # Download VOC 2012 dataset
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    download_path = Path(DOWNLOAD_DIR) / 'VOCtrainval_11-May-2012.tar'
    if download_path.exists():
        print(f'Download path {download_path} already exists! download step is skipped.')
    else:
        torch.hub.download_url_to_file(VOC2012_URL, download_path)

    # Extract tar file
    ap = tarfile.open(download_path)

    voc2012_path = Path(args.dir) / 'voc2012_seg'
    os.makedirs(voc2012_path, exist_ok=True)
    ap.extractall(voc2012_path)
    ap.close()

    img_src = voc2012_path / 'VOCdevkit' / 'VOC2012' / 'JPEGImages'
    label_src = voc2012_path / 'VOCdevkit' / 'VOC2012' / 'SegmentationClass'

    # Get train sample list
    train_samples = voc2012_path / 'VOCdevkit' / 'VOC2012' / 'ImageSets' / 'Segmentation' / 'train.txt'
    with open(train_samples, 'r') as f:
        train_samples = f.readlines()
        train_samples = [sample.strip() for sample in train_samples]

    # Move train images and masks
    train_image_dir = voc2012_path / 'images' / 'train'
    train_label_dir = voc2012_path / 'labels' / 'train'

    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    for sample in tqdm(train_samples):
        shutil.move(img_src / (sample + '.jpg'), train_image_dir / (sample + '.jpg'))
        label = Image.open(label_src / (sample + '.png'))
        if args.label_image_mode == 'L':
            label = np.array(label)
            label = Image.fromarray(label, mode='L')
        elif args.label_image_mode == 'RGB':
            label = label.convert('RGB')
        elif args.label_image_mode == 'P':
            pass
        label.save(train_label_dir / (sample + '.png'))

    # Get valid sample list
    valid_samples = voc2012_path / 'VOCdevkit' / 'VOC2012' / 'ImageSets' / 'Segmentation' / 'val.txt'
    with open(valid_samples, 'r') as f:
        valid_samples = f.readlines()
        valid_samples = [sample.strip() for sample in valid_samples]

    # Move valid images and masks
    valid_image_dir = voc2012_path / 'images' / 'valid'
    valid_label_dir = voc2012_path / 'labels' / 'valid'

    os.makedirs(valid_image_dir, exist_ok=True)
    os.makedirs(valid_label_dir, exist_ok=True)
    for sample in tqdm(valid_samples):
        shutil.move(img_src / (sample + '.jpg'), valid_image_dir / (sample + '.jpg'))
        if args.label_image_mode == 'L':
            label = Image.open(label_src / (sample + '.png'))
            label = np.array(label)
            label = Image.fromarray(label, mode='L')
            label.save(valid_label_dir / (sample + '.png'))
        else:
            Image.open(label_src / (sample + '.png')).save(valid_label_dir / (sample + '.png'))

    # Build id_mapping
    if args.label_image_mode == 'L':
        id_mapping_to_save = list(VOC2012_ID_MAPPING_RGB.values())
    else:
        id_mapping_to_save = VOC2012_ID_MAPPING_RGB

    with open(voc2012_path / 'id_mapping.json', 'w') as f:
        json.dump(id_mapping_to_save, f)

    try:
        shutil.rmtree(voc2012_path / 'VOCdevkit')
    except OSError as e:
        print(e)
