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
WFLW_IMAGES_URL = 'https://drive.usercontent.google.com/download?id=1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC&export=download&authuser=1&confirm=t&uuid=a62cb82a-66a0-498c-b568-5b1955f3926d&at=APZUnTX3W3OXP2Y2kHd4OpGltbjL%3A1714472372882'
WFLW_ANNOTATIONS_URL = 'https://wywu.github.io/projects/LAB/support/WFLW_annotations.tar.gz'
KEYPOINT_INFO = {
        0: dict(name='kpt-0', id=0, color=[255, 0, 0], type='', swap='kpt-32'),
        1: dict(name='kpt-1', id=1, color=[255, 0, 0], type='', swap='kpt-31'),
        2: dict(name='kpt-2', id=2, color=[255, 0, 0], type='', swap='kpt-30'),
        3: dict(name='kpt-3', id=3, color=[255, 0, 0], type='', swap='kpt-29'),
        4: dict(name='kpt-4', id=4, color=[255, 0, 0], type='', swap='kpt-28'),
        5: dict(name='kpt-5', id=5, color=[255, 0, 0], type='', swap='kpt-27'),
        6: dict(name='kpt-6', id=6, color=[255, 0, 0], type='', swap='kpt-26'),
        7: dict(name='kpt-7', id=7, color=[255, 0, 0], type='', swap='kpt-25'),
        8: dict(name='kpt-8', id=8, color=[255, 0, 0], type='', swap='kpt-24'),
        9: dict(name='kpt-9', id=9, color=[255, 0, 0], type='', swap='kpt-23'),
        10: dict(name='kpt-10', id=10, color=[255, 0, 0], type='', swap='kpt-22'),
        11: dict(name='kpt-11', id=11, color=[255, 0, 0], type='', swap='kpt-21'),
        12: dict(name='kpt-12', id=12, color=[255, 0, 0], type='', swap='kpt-20'),
        13: dict(name='kpt-13', id=13, color=[255, 0, 0], type='', swap='kpt-19'),
        14: dict(name='kpt-14', id=14, color=[255, 0, 0], type='', swap='kpt-18'),
        15: dict(name='kpt-15', id=15, color=[255, 0, 0], type='', swap='kpt-17'),
        16: dict(name='kpt-16', id=16, color=[255, 0, 0], type='', swap=''),
        17: dict(name='kpt-17', id=17, color=[255, 0, 0], type='', swap='kpt-15'),
        18: dict(name='kpt-18', id=18, color=[255, 0, 0], type='', swap='kpt-14'),
        19: dict(name='kpt-19', id=19, color=[255, 0, 0], type='', swap='kpt-13'),
        20: dict(name='kpt-20', id=20, color=[255, 0, 0], type='', swap='kpt-12'),
        21: dict(name='kpt-21', id=21, color=[255, 0, 0], type='', swap='kpt-11'),
        22: dict(name='kpt-22', id=22, color=[255, 0, 0], type='', swap='kpt-10'),
        23: dict(name='kpt-23', id=23, color=[255, 0, 0], type='', swap='kpt-9'),
        24: dict(name='kpt-24', id=24, color=[255, 0, 0], type='', swap='kpt-8'),
        25: dict(name='kpt-25', id=25, color=[255, 0, 0], type='', swap='kpt-7'),
        26: dict(name='kpt-26', id=26, color=[255, 0, 0], type='', swap='kpt-6'),
        27: dict(name='kpt-27', id=27, color=[255, 0, 0], type='', swap='kpt-5'),
        28: dict(name='kpt-28', id=28, color=[255, 0, 0], type='', swap='kpt-4'),
        29: dict(name='kpt-29', id=29, color=[255, 0, 0], type='', swap='kpt-3'),
        30: dict(name='kpt-30', id=30, color=[255, 0, 0], type='', swap='kpt-2'),
        31: dict(name='kpt-31', id=31, color=[255, 0, 0], type='', swap='kpt-1'),
        32: dict(name='kpt-32', id=32, color=[255, 0, 0], type='', swap='kpt-0'),
        33: dict(name='kpt-33', id=33, color=[255, 0, 0], type='', swap='kpt-46'),
        34: dict(name='kpt-34', id=34, color=[255, 0, 0], type='', swap='kpt-45'),
        35: dict(name='kpt-35', id=35, color=[255, 0, 0], type='', swap='kpt-44'),
        36: dict(name='kpt-36', id=36, color=[255, 0, 0], type='', swap='kpt-43'),
        37: dict(name='kpt-37', id=37, color=[255, 0, 0], type='', swap='kpt-42'),
        38: dict(name='kpt-38', id=38, color=[255, 0, 0], type='', swap='kpt-50'),
        39: dict(name='kpt-39', id=39, color=[255, 0, 0], type='', swap='kpt-49'),
        40: dict(name='kpt-40', id=40, color=[255, 0, 0], type='', swap='kpt-48'),
        41: dict(name='kpt-41', id=41, color=[255, 0, 0], type='', swap='kpt-47'),
        42: dict(name='kpt-42', id=42, color=[255, 0, 0], type='', swap='kpt-37'),
        43: dict(name='kpt-43', id=43, color=[255, 0, 0], type='', swap='kpt-36'),
        44: dict(name='kpt-44', id=44, color=[255, 0, 0], type='', swap='kpt-35'),
        45: dict(name='kpt-45', id=45, color=[255, 0, 0], type='', swap='kpt-34'),
        46: dict(name='kpt-46', id=46, color=[255, 0, 0], type='', swap='kpt-33'),
        47: dict(name='kpt-47', id=47, color=[255, 0, 0], type='', swap='kpt-41'),
        48: dict(name='kpt-48', id=48, color=[255, 0, 0], type='', swap='kpt-40'),
        49: dict(name='kpt-49', id=49, color=[255, 0, 0], type='', swap='kpt-39'),
        50: dict(name='kpt-50', id=50, color=[255, 0, 0], type='', swap='kpt-38'),
        51: dict(name='kpt-51', id=51, color=[255, 0, 0], type='', swap=''),
        52: dict(name='kpt-52', id=52, color=[255, 0, 0], type='', swap=''),
        53: dict(name='kpt-53', id=53, color=[255, 0, 0], type='', swap=''),
        54: dict(name='kpt-54', id=54, color=[255, 0, 0], type='', swap=''),
        55: dict(name='kpt-55', id=55, color=[255, 0, 0], type='', swap='kpt-59'),
        56: dict(name='kpt-56', id=56, color=[255, 0, 0], type='', swap='kpt-58'),
        57: dict(name='kpt-57', id=57, color=[255, 0, 0], type='', swap=''),
        58: dict(name='kpt-58', id=58, color=[255, 0, 0], type='', swap='kpt-56'),
        59: dict(name='kpt-59', id=59, color=[255, 0, 0], type='', swap='kpt-55'),
        60: dict(name='kpt-60', id=60, color=[255, 0, 0], type='', swap='kpt-72'),
        61: dict(name='kpt-61', id=61, color=[255, 0, 0], type='', swap='kpt-71'),
        62: dict(name='kpt-62', id=62, color=[255, 0, 0], type='', swap='kpt-70'),
        63: dict(name='kpt-63', id=63, color=[255, 0, 0], type='', swap='kpt-69'),
        64: dict(name='kpt-64', id=64, color=[255, 0, 0], type='', swap='kpt-68'),
        65: dict(name='kpt-65', id=65, color=[255, 0, 0], type='', swap='kpt-75'),
        66: dict(name='kpt-66', id=66, color=[255, 0, 0], type='', swap='kpt-74'),
        67: dict(name='kpt-67', id=67, color=[255, 0, 0], type='', swap='kpt-73'),
        68: dict(name='kpt-68', id=68, color=[255, 0, 0], type='', swap='kpt-64'),
        69: dict(name='kpt-69', id=69, color=[255, 0, 0], type='', swap='kpt-63'),
        70: dict(name='kpt-70', id=70, color=[255, 0, 0], type='', swap='kpt-62'),
        71: dict(name='kpt-71', id=71, color=[255, 0, 0], type='', swap='kpt-61'),
        72: dict(name='kpt-72', id=72, color=[255, 0, 0], type='', swap='kpt-60'),
        73: dict(name='kpt-73', id=73, color=[255, 0, 0], type='', swap='kpt-67'),
        74: dict(name='kpt-74', id=74, color=[255, 0, 0], type='', swap='kpt-66'),
        75: dict(name='kpt-75', id=75, color=[255, 0, 0], type='', swap='kpt-65'),
        76: dict(name='kpt-76', id=76, color=[255, 0, 0], type='', swap='kpt-82'),
        77: dict(name='kpt-77', id=77, color=[255, 0, 0], type='', swap='kpt-81'),
        78: dict(name='kpt-78', id=78, color=[255, 0, 0], type='', swap='kpt-80'),
        79: dict(name='kpt-79', id=79, color=[255, 0, 0], type='', swap=''),
        80: dict(name='kpt-80', id=80, color=[255, 0, 0], type='', swap='kpt-78'),
        81: dict(name='kpt-81', id=81, color=[255, 0, 0], type='', swap='kpt-77'),
        82: dict(name='kpt-82', id=82, color=[255, 0, 0], type='', swap='kpt-76'),
        83: dict(name='kpt-83', id=83, color=[255, 0, 0], type='', swap='kpt-87'),
        84: dict(name='kpt-84', id=84, color=[255, 0, 0], type='', swap='kpt-86'),
        85: dict(name='kpt-85', id=85, color=[255, 0, 0], type='', swap=''),
        86: dict(name='kpt-86', id=86, color=[255, 0, 0], type='', swap='kpt-84'),
        87: dict(name='kpt-87', id=87, color=[255, 0, 0], type='', swap='kpt-83'),
        88: dict(name='kpt-88', id=88, color=[255, 0, 0], type='', swap='kpt-92'),
        89: dict(name='kpt-89', id=89, color=[255, 0, 0], type='', swap='kpt-91'),
        90: dict(name='kpt-90', id=90, color=[255, 0, 0], type='', swap=''),
        91: dict(name='kpt-91', id=91, color=[255, 0, 0], type='', swap='kpt-89'),
        92: dict(name='kpt-92', id=92, color=[255, 0, 0], type='', swap='kpt-88'),
        93: dict(name='kpt-93', id=93, color=[255, 0, 0], type='', swap='kpt-95'),
        94: dict(name='kpt-94', id=94, color=[255, 0, 0], type='', swap=''),
        95: dict(name='kpt-95', id=95, color=[255, 0, 0], type='', swap='kpt-93'),
        96: dict(name='kpt-96', id=96, color=[255, 0, 0], type='', swap='kpt-97'),
        97: dict(name='kpt-97', id=97, color=[255, 0, 0], type='', swap='kpt-96')
    }


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':
    # Set argument (data directory)
    parser = argparse.ArgumentParser(description="Parser for WFLW dataset downloader.")
    parser.add_argument('--dir', type=str, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    # Download wflw dataset
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    download_path = Path(DOWNLOAD_DIR) / 'WFLW_images.tar.gz'
    if download_path.exists():
        print(f'Download path {download_path} already exists! download step is skipped.')
    else:
        torch.hub.download_url_to_file(WFLW_IMAGES_URL, download_path)

    download_path = Path(DOWNLOAD_DIR) / 'WFLW_annotations.tar.gz'
    if download_path.exists():
        print(f'Download path {download_path} already exists! download step is skipped.')
    else:
        torch.hub.download_url_to_file(WFLW_ANNOTATIONS_URL, download_path)

    # Set base directory
    wflw_path = Path(args.dir) / 'wflw'
    os.makedirs(wflw_path, exist_ok=True)

    # Extract images
    ap = tarfile.open(Path(DOWNLOAD_DIR) / 'WFLW_images.tar.gz')
    ap.extractall(wflw_path)
    ap.close()

    # Extract annotations
    ap = tarfile.open(Path(DOWNLOAD_DIR) / 'WFLW_annotations.tar.gz')
    ap.extractall(wflw_path)
    ap.close()

    ori_images_root = wflw_path / 'WFLW_images'
    ori_ann_root = wflw_path / 'WFLW_annotations'

    # Train
    phase = 'train'

    train_image_dir = wflw_path / 'images' / 'train'
    train_label_dir = wflw_path / 'labels' / 'train'
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)

    with open(ori_ann_root / 'list_98pt_rect_attr_train_test' / 'list_98pt_rect_attr_train.txt') as f:
        train_lines = f.readlines()

    train_dict = {}
    for line in train_lines:
        line = line.strip()
        ann = line.split(' ')

        keypoints = ann[:-11]
        box = ann[-11:-7]
        attributes = ann[-7:-1]
        img = ann[-1]

        keypoints = np.array(keypoints)
        keypoints = keypoints.reshape(-1, 2)
        keypoints = np.hstack((keypoints, np.ones((len(keypoints), 1))))
        keypoints = keypoints.ravel()
        keypoints = keypoints.tolist()

        if img not in train_dict:
            train_dict[img] = []
        
        sample = ' '.join(keypoints)
        sample += ' ' + ' '.join(box)

        train_dict[img].append(sample)

    for key, val in train_dict.items():
        src_img = key
        img = Path(key.split('/')[-1])
        label = Path(img.stem + '.txt')
        sample = '\n'.join(val)

        shutil.copy(ori_images_root / src_img, train_image_dir / img)
        with open(train_label_dir / label, 'w') as f:
            f.write(sample)
            f.close()

    # Validation
    phase = 'valid'

    valid_image_dir = wflw_path / 'images' / 'valid'
    valid_label_dir = wflw_path / 'labels' / 'valid'
    os.makedirs(valid_image_dir, exist_ok=True)
    os.makedirs(valid_label_dir, exist_ok=True)
    
    with open(ori_ann_root / 'list_98pt_rect_attr_train_test' / 'list_98pt_rect_attr_test.txt') as f:
        val_lines = f.readlines()

    val_dict = {}
    for line in val_lines:
        line = line.strip()
        ann = line.split(' ')

        keypoints = ann[:-11]
        box = ann[-11:-7]
        attributes = ann[-7:-1]
        img = ann[-1]

        keypoints = np.array(keypoints)
        keypoints = keypoints.reshape(-1, 2)
        keypoints = np.hstack((keypoints, np.ones((len(keypoints), 1))))
        keypoints = keypoints.ravel()
        keypoints = keypoints.tolist()

        if img not in val_dict:
            val_dict[img] = []
        
        sample = ' '.join(keypoints)
        sample += ' ' + ' '.join(box)

        val_dict[img].append(sample)

    for key, val in val_dict.items():
        src_img = key
        img = Path(key.split('/')[-1])
        label = Path(img.stem + '.txt')
        sample = '\n'.join(val)

        shutil.copy(ori_images_root / src_img, valid_image_dir / img)
        with open(valid_label_dir / label, 'w') as f:
            f.write(sample)
            f.close()

    # Build id_mapping 
    id_mapping = []
    for key, val in KEYPOINT_INFO.items():
        swap = val['swap']
        if swap == '':
            swap = val['name']
        
        id_mapping.append(dict(name=val['name'], skeleton=None, swap=swap))

    with open(wflw_path / 'id_mapping.json', 'w') as f:
        json.dump(id_mapping, f)

    try:
        shutil.rmtree(ori_images_root)
        shutil.rmtree(ori_ann_root)
    except OSError as e:
        print(e)
