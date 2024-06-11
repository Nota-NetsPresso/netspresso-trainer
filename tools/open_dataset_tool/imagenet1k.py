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
DOWNLOAD_DIR = './data/download'
CLASS_INDEX_URL = 'https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/metadata/imagenet_class_index.json'


if __name__ == '__main__':
    # Set argument (data directory)
    parser = argparse.ArgumentParser(description="Parser for ImageNet1K dataset remapper.")
    parser.add_argument('--dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--train-images', type=str, dest='train_images', required=True,
                        help="ImageNet1K dataset cannot be downloaded automatically. Download dataset from https://www.image-net.org/ and set train path here.")
    parser.add_argument('--valid-images', type=str, dest='valid_images', required=True,
                        help="ImageNet1K dataset cannot be downloaded automatically. Download dataset from https://www.image-net.org/ and set validation path here.")
    parser.add_argument('--devkit', type=str, dest='devkit', required=True,
                        help="ImageNet1K dataset cannot be downloaded automatically. Download dataset from https://www.image-net.org/ and set devkit path here.")
    args = parser.parse_args()

    # Download class index
    class_index_path = Path(DOWNLOAD_DIR) / 'imagenet_class_index.json'
    if class_index_path.exists():
        print(f'Download path {class_index_path} already exists! download step is skipped.')
    else:
        torch.hub.download_url_to_file(CLASS_INDEX_URL, class_index_path)

    # Set base directory
    imagenet_path = Path(args.dir) / 'imagenet1k'
    os.makedirs(imagenet_path, exist_ok=True)

    # Extract train split
    train_image_dir = imagenet_path / 'images' / 'train'
    os.makedirs(train_image_dir, exist_ok=True)

    print('Extracting training images tar file... This may take a minutes.')
    tmp2extract = imagenet_path / 'tmp2extract'
    os.makedirs(tmp2extract, exist_ok=True)

    ap = tarfile.open(Path(args.train_images))
    ap.extractall(tmp2extract)
    ap.close()

    print('Extracting training images from each class tar file.')
    tar_files = os.listdir(tmp2extract)
    for extract_file in tqdm(tar_files):
        extract_file = tmp2extract / (extract_file)
        ap = tarfile.open(extract_file)
        ap.extractall(train_image_dir)
        ap.close()
        try:
            os.remove(extract_file)
        except OSError as e:
            print(e)
    try:
        shutil.rmtree(tmp2extract)
    except OSError as e:
        print(e)
    print('Done!')

    # Extract valid split
    print('Extracting valid images tar file...')
    valid_image_dir = imagenet_path / 'images' / 'valid'
    os.makedirs(valid_image_dir, exist_ok=True)

    ap = tarfile.open(Path(args.valid_images))
    ap.extractall(valid_image_dir) # Extract directly
    ap.close()

    print('Done!')

    # Extract meta files
    print('Extracting meta data ...')
    ap = tarfile.open(Path(args.devkit))
    ap.extractall(imagenet_path)
    ap.close()

    devkit_extracted = imagenet_path / 'ILSVRC2012_devkit_t12'
    meta = devkit_extracted / 'data' / 'meta.mat'

    mat = scipy.io.loadmat(meta)

    base_nids = []
    for row in mat['synsets']:
        row = row[0]
        cls_nid = row[1].item()
        base_nids.append(cls_nid)

    with open(class_index_path, 'r') as f:
        json_object = json.load(f)
        nid_to_label = {json_object[str(i)][0]:i for i in range(1000)}
        class_names = [json_object[str(i)][1] for i in range(1000)]

    print('Done!')

    # Build train label csv file
    print('Building train label csv file ...')
    train_label_dir = imagenet_path / 'labels'
    os.makedirs(train_label_dir, exist_ok=True)

    samples = os.listdir(imagenet_path / 'images' / 'train')
    labels = []
    for sample in samples:
        labels.append(nid_to_label[sample.split('_')[0]])
    
    train_csv = pd.DataFrame({'image_id': samples, 'class': labels})
    train_csv.to_csv(train_label_dir / 'imagenet_train.csv', mode='w', index=False)
    print('Done!')

    # Build valid label csv file
    print('Building valid label csv file ...')
    valid_label_dir = imagenet_path / 'labels'
    os.makedirs(valid_label_dir, exist_ok=True)

    with open(devkit_extracted / 'data' / 'ILSVRC2012_validation_ground_truth.txt') as f:
        labels = f.readlines()
        labels = [int(l.strip()) - 1 for l in labels]
        f.close()

    valid_nids = [base_nids[label] for label in labels]
    valid_labels = [nid_to_label[nid] for nid in valid_nids]
    valid_csv = pd.DataFrame({'image_id': sorted(os.listdir(imagenet_path / 'images' / 'valid')), 'class': valid_labels})
    valid_csv.to_csv(valid_label_dir / 'imagenet_valid.csv', mode='w', index=False)
    print('Done!')

    try:
        shutil.rmtree(devkit_extracted)
    except OSError as e:
        print(e)

    # Build id_mapping
    print('Building id_mapping ...')
    id_mapping = class_names
    with open(imagenet_path / 'id_mapping.json', 'w') as f:
        json.dump(id_mapping, f)

    print('Done!')
