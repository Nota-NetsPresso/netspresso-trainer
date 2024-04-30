import argparse
from pathlib import Path
import os
import tarfile
import shutil

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

DEFAULT_DATA_DIR = './data'
DOWNLOAD_DIR = './data/download'
VOC2012_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':
    # Set argument (data directory)
    parser = argparse.ArgumentParser(description="Parser for VOC 2012 dataset downloader.")
    parser.add_argument('--dir', type=str, default=DEFAULT_DATA_DIR)
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
    for sample in train_samples:
        shutil.move(img_src / (sample + '.jpg'), train_image_dir / (sample + '.jpg'))
        shutil.move(label_src / (sample + '.png'), train_label_dir / (sample + '.png'))
    
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
    for sample in valid_samples:
        shutil.move(img_src / (sample + '.jpg'), valid_image_dir / (sample + '.jpg'))
        shutil.move(label_src / (sample + '.png'), valid_label_dir / (sample + '.png'))
    
    try:
        shutil.rmtree(voc2012_path / 'VOCdevkit')
    except OSError as e:
        print(e)
