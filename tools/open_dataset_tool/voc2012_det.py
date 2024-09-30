import argparse
import os
from pathlib import Path
import tarfile
import json

from tqdm import tqdm
import scipy.io
import xml.etree.ElementTree as ET
import shutil
import torch

DEFAULT_DATA_DIR = './data'
DOWNLOAD_DIR = './data/download'
VOC2012_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            'dog', 'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']
CLASS_DICT = {}
for i, c in enumerate(CLASSES):
    CLASS_DICT[c] = i


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

    voc2012_path = Path(args.dir) / 'voc2012_det'
    os.makedirs(voc2012_path, exist_ok=True)
    ap.extractall(voc2012_path)
    ap.close()

    extracted_root = voc2012_path / 'VOCdevkit' / 'VOC2012'
    train_list = extracted_root / 'ImageSets' / 'Main' / 'train.txt'
    with open(train_list, 'r') as f:
        train_list = f.readlines()
        train_list = [x.strip() for x in train_list]
        f.close()
    val_list = extracted_root / 'ImageSets' / 'Main' / 'val.txt'
    with open(val_list, 'r') as f:
        val_list = f.readlines()
        val_list = [x.strip() for x in val_list]
        f.close()

    image_src = extracted_root / 'JPEGImages'
    label_src = extracted_root / 'Annotations'

    # Set training data
    train_image_dir = voc2012_path / 'images' / 'train'
    train_label_dir = voc2012_path / 'labels' / 'train'

    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)

    for sample_name in tqdm(train_list):
        image_path = image_src / (sample_name + '.jpg')
        label_path = label_src / (sample_name + '.xml')

        label = ET.parse(label_path).getroot()
        label_txt = ''
        size = label.find('size')
        h, w = int(size.find('height').text), int(size.find('width').text)

        for item in label.iter('object'):
            class_name = item.find('name').text
            label = CLASS_DICT[class_name]

            box = item.find('bndbox')
            xmax = int(box.find('xmax').text.split('.')[0])
            xmin = int(box.find('xmin').text.split('.')[0])
            ymax = int(box.find('ymax').text.split('.')[0])
            ymin = int(box.find('ymin').text.split('.')[0])

            cx = (xmax + xmin) / 2.0 / w
            cy = (ymax + ymin) / 2.0 / h
            box_w = (xmax - xmin) * 1.0 / w
            box_h = (ymax - ymin) * 1.0 / h

            label_txt += f'{label} {cx} {cy} {box_w} {box_h}\n'

        target_image_path = train_image_dir / (sample_name + '.jpg')
        target_ann_path = train_label_dir / (sample_name + '.txt')

        with open(target_ann_path, 'w') as f:
            f.write(label_txt)
            f.close()
        shutil.copy(image_path, target_image_path)

    # Set validation data
    valid_image_dir = voc2012_path / 'images' / 'valid'
    valid_label_dir = voc2012_path / 'labels' / 'valid'

    os.makedirs(valid_image_dir, exist_ok=True)
    os.makedirs(valid_label_dir, exist_ok=True)

    for sample_name in tqdm(val_list):
        image_path = image_src / (sample_name + '.jpg')
        label_path = label_src / (sample_name + '.xml')

        label = ET.parse(label_path).getroot()
        label_txt = ''
        size = label.find('size')
        h, w = int(size.find('height').text), int(size.find('width').text)

        for item in label.iter('object'):
            class_name = item.find('name').text
            label = CLASS_DICT[class_name]

            box = item.find('bndbox')
            xmax = int(box.find('xmax').text.split('.')[0])
            xmin = int(box.find('xmin').text.split('.')[0])
            ymax = int(box.find('ymax').text.split('.')[0])
            ymin = int(box.find('ymin').text.split('.')[0])

            cx = (xmax + xmin) / 2.0 / w
            cy = (ymax + ymin) / 2.0 / h
            box_w = (xmax - xmin) * 1.0 / w
            box_h = (ymax - ymin) * 1.0 / h

            label_txt += f'{label} {cx} {cy} {box_w} {box_h}\n'

        target_image_path = valid_image_dir / (sample_name + '.jpg')
        target_ann_path = valid_label_dir / (sample_name + '.txt')

        with open(target_ann_path, 'w') as f:
            f.write(label_txt)
            f.close()
        shutil.copy(image_path, target_image_path)

    # Build id_mapping
    with open(voc2012_path / 'id_mapping.json', 'w') as f:
        json.dump(CLASSES, f)

    # Clean up
    shutil.rmtree(voc2012_path / 'VOCdevkit')
