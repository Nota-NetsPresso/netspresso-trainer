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

import scipy
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

DEFAULT_DATA_DIR = './data'
CITYSCAPES_CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                      'traffic light', 'traffic sign', 'vegetation', 'terrain',
                      'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                      'motorcycle', 'bicycle')
# Map labels according to https://github.com/mcordts/cityscapesScripts/blob/cf14a15f14bb868c5a9f14acab5ef3120b97df32/cityscapesscripts/helpers/labels.py#L62-L99
CITYSCAPES_LABELS = np.array([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33])
MAPPING_TABLE = np.zeros(35).astype('uint8') + 255
MAPPING_TABLE[CITYSCAPES_LABELS] = np.arange(19)


if __name__ == '__main__':
    # Set argument (data directory)
    parser = argparse.ArgumentParser(description="Parser for Cityscapes dataset remapper.")
    parser.add_argument('--dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--images', type=str, dest='images', required=True,
                        help="Cityscapes dataset cannot be downloaded automatically. Download dataset from https://www.cityscapes-dataset.com/ and set leftimg8bit_trainvaltest.zip path here.")
    parser.add_argument('--labels', type=str, dest='labels', required=True,
                        help="Cityscapes dataset cannot be downloaded automatically. Download dataset from https://www.cityscapes-dataset.com/ and set gtFine_trainvaltest.zip path here.")
    args = parser.parse_args()

    # Set base directory
    cityscapes_path = Path(args.dir) / 'cityscapes'
    os.makedirs(cityscapes_path, exist_ok=True)

    # Extract images
    print('Extracting images zip file... This may take a minutes.')
    tmp2extract = cityscapes_path / 'tmp2extract'
    shutil.unpack_archive(args.images, tmp2extract, "zip")

    # Set image directories
    train_image_dir = cityscapes_path / 'images' / 'train'
    valid_image_dir = cityscapes_path / 'images' / 'valid'
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(valid_image_dir, exist_ok=True)

    print('Constructing image directories...')
    train_img_src = tmp2extract / 'leftImg8bit' / 'train'
    for path, dir, files in os.walk(train_img_src):
        for file in files:
            shutil.move(Path(path) / file, train_image_dir / file.replace('_leftImg8bit', ''))

    valid_img_src = tmp2extract / 'leftImg8bit' / 'val'
    for path, dir, files in os.walk(valid_img_src):
        for file in files:
            shutil.move(Path(path) / file, valid_image_dir / file.replace('_leftImg8bit', ''))

    # Extract labels
    print('Extracting labels zip file... This may take a minutes.')
    tmp2extract = cityscapes_path / 'tmp2extract'
    shutil.unpack_archive(args.labels, tmp2extract, "zip")

    # Set label directories
    train_label_dir = cityscapes_path / 'labels' / 'train'
    valid_label_dir = cityscapes_path / 'labels' / 'valid'
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(valid_label_dir, exist_ok=True)

    print('Constructing label directories...')
    train_label_src = tmp2extract / 'gtFine' / 'train'
    for path, dir, files in os.walk(train_label_src):
        for file in files:
            if 'labelIds' in file:
                label = cv2.imread(str(Path(path) / file), cv2.IMREAD_GRAYSCALE)
                label = MAPPING_TABLE[label]
                cv2.imwrite(str(train_label_dir / file.replace('_gtFine_labelIds', '')), label)

    valid_label_src = tmp2extract / 'gtFine' / 'val'
    for path, dir, files in os.walk(valid_label_src):
        for file in files:
            if 'labelIds' in file:
                label = cv2.imread(str(Path(path) / file), cv2.IMREAD_GRAYSCALE)
                label = MAPPING_TABLE[label]
                cv2.imwrite(str(valid_label_dir / file.replace('_gtFine_labelIds', '')), label)

    # Build id_mapping
    with open(cityscapes_path / 'id_mapping.json', 'w') as f:
        json.dump(CITYSCAPES_CLASSES, f)

    # Remove temporary directory
    try:
        shutil.rmtree(tmp2extract)
    except OSError as e:
        print(e)
