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
CIFAR100_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
CIFAR100_CLASSES = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':
    # Set argument (data directory)
    parser = argparse.ArgumentParser(description="Parser for CIFAR100 dataset downloader.")
    parser.add_argument('--dir', type=str, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    # Download cifar 100 dataset
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    download_path = Path(DOWNLOAD_DIR) / 'cifar100.tar.gz'
    if download_path.exists():
        print(f'Download path {download_path} already exists! download step is skipped.')
    else:
        torch.hub.download_url_to_file(CIFAR100_URL, download_path)

    # Extract tar.gz file
    ap = tarfile.open(download_path)

    data_dir = Path(args.dir) / 'cifar100'
    os.makedirs(data_dir, exist_ok=True)
    ap.extractall(data_dir)
    ap.close()

    # Read data
    extracted_dir = data_dir / 'cifar-100-python' # auto-generated by ap.extractall
    cifar_train = extracted_dir / 'train'
    cifar_test = extracted_dir / 'test'
    cifar_meta = extracted_dir / 'meta'

    cifar_train = unpickle(cifar_train)
    cifar_test = unpickle(cifar_test)
    cifar_meta = unpickle(cifar_meta)

    cifar_meta = cifar_meta[b'fine_label_names']
    cifar_meta = [c.decode('utf-8') for c in cifar_meta]

    # Re-format train split
    train_image_dir = data_dir / 'images' / 'train'
    train_label_dir = data_dir / 'labels'

    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)

    train_names = cifar_train[b'filenames']
    train_names = [n.decode('utf-8') for n in train_names]
    train_images = cifar_train[b'data']
    for name, image in tqdm(zip(train_names, train_images)):
        image = np.transpose(image.reshape(3, 32, 32), (1, 2, 0))
        image = image[..., ::-1]
        cv2.imwrite(str(train_image_dir / name), image)

    train_labels = cifar_train[b'fine_labels']
    train_label_csv = pd.DataFrame({'image_id': train_names, 'class': train_labels})
    train_label_csv.to_csv(train_label_dir / 'cifar100_train.csv', mode='w', index=False)

    # Re-format valid split
    valid_image_dir = data_dir / 'images' / 'valid'
    valid_label_dir = data_dir / 'labels'

    os.makedirs(valid_image_dir, exist_ok=True)
    os.makedirs(valid_label_dir, exist_ok=True)

    val_names = cifar_test[b'filenames']
    val_names = [n.decode('utf-8') for n in val_names]
    val_images = cifar_test[b'data']
    for name, image in tqdm(zip(val_names, val_images)):
        image = np.transpose(image.reshape(3, 32, 32), (1, 2, 0))
        image = image[..., ::-1]
        cv2.imwrite(str(valid_image_dir / name), image)

    val_labels = cifar_test[b'fine_labels']
    val_label_csv = pd.DataFrame({'image_id': val_names, 'class': val_labels})
    val_label_csv.to_csv(valid_label_dir / 'cifar100_val.csv', mode='w', index=False)

    # Build id_mapping
    with open(data_dir / 'id_mapping.json', 'w') as f:
        json.dump(CIFAR100_CLASSES, f)

    try:
        shutil.rmtree(extracted_dir)
    except OSError as e:
        print(e)