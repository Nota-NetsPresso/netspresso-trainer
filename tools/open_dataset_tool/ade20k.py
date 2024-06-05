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
import shutil
import json

import numpy as np
from PIL import Image
import torch

DEFAULT_DATA_DIR = './data'
DOWNLOAD_DIR = './data/download'
ADE20K_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
ID_MAPPING = ['wall', 'building;edifice', 'sky', 'floor;flooring', 'tree', 'ceiling', 'skyscraper', 'bed', 'windowpane;window', 'grass', 'cabinet', 'sidewalk;pavement', 'person;individual;someone;somebody;mortal;soul', 'earth;ground', 'door;double;door', 'table', 'mountain;mount', 'plant;flora;plant;life', 'curtain;drape;drapery;mantle;pall', 'chair', 'car;auto;automobile;machine;motorcar', 'water', 'painting;picture', 'sofa;couch;lounge', 'shelf', 'house', 'sea', 'mirror', 'rug;carpet;carpeting', 'field', 'armchair', 'seat', 'fence;fencing', 'desk', 'rock;stone', 'wardrobe;closet;press', 'lamp', 'bathtub;bathing;tub;bath;tub', 'railing;rail', 'cushion', 'base;pedestal;stand', 'box', 'column;pillar', 'signboard;sign', 'chest;of;drawers;chest;bureau;dresser', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace;hearth;open;fireplace', 'refrigerator;icebox', 'grandstand;covered;stand', 'path', 'stairs;steps', 'runway', 'case;display;case;showcase;vitrine', 'pool;table;billiard;table;snooker;table', 'pillow', 'screen;door;screen', 'stairway;staircase', 'river', 'bridge;span', 'bookcase', 'blind;screen', 'coffee;table;cocktail;table', 'toilet;can;commode;crapper;pot;potty;stool;throne', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove;kitchen;stove;range;kitchen;range;cooking;stove', 'palm;palm;tree', 'kitchen;island', 'computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system', 'swivel;chair', 'boat', 'bar', 'arcade;machine', 'hovel;hut;hutch;shack;shanty', 'bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle', 'towel', 'light;light;source', 'truck;motortruck', 'tower', 'chandelier;pendant;pendent', 'awning;sunshade;sunblind', 'streetlight;street;lamp', 'booth;cubicle;stall;kiosk', 'television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box', 'airplane;aeroplane;plane', 'dirt;track', 'apparel;wearing;apparel;dress;clothes', 'pole', 'land;ground;soil', 'bannister;banister;balustrade;balusters;handrail', 'escalator;moving;staircase;moving;stairway', 'ottoman;pouf;pouffe;puff;hassock', 'bottle', 'buffet;counter;sideboard', 'poster;posting;placard;notice;bill;card', 'stage', 'van', 'ship', 'fountain', 'conveyer;belt;conveyor;belt;conveyer;conveyor;transporter', 'canopy', 'washer;automatic;washer;washing;machine', 'plaything;toy', 'swimming;pool;swimming;bath;natatorium', 'stool', 'barrel;cask', 'basket;handbasket', 'waterfall;falls', 'tent;collapsible;shelter', 'bag', 'minibike;motorbike', 'cradle', 'oven', 'ball', 'food;solid;food', 'step;stair', 'tank;storage;tank', 'trade;name;brand;name;brand;marque', 'microwave;microwave;oven', 'pot;flowerpot', 'animal;animate;being;beast;brute;creature;fauna', 'bicycle;bike;wheel;cycle', 'lake', 'dishwasher;dish;washer;dishwashing;machine', 'screen;silver;screen;projection;screen', 'blanket;cover', 'sculpture', 'hood;exhaust;hood', 'sconce', 'vase', 'traffic;light;traffic;signal;stoplight', 'tray', 'ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin', 'fan', 'pier;wharf;wharfage;dock', 'crt;screen', 'plate', 'monitor;monitoring;device', 'bulletin;board;notice;board', 'shower', 'radiator', 'glass;drinking;glass', 'clock', 'flag']


if __name__ == '__main__':
    # Set argument (data directory)
    parser = argparse.ArgumentParser(description="Parser for ADE20K dataset remapper.")
    parser.add_argument('--dir', type=str, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    # Download ade20k dataset
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    download_path = Path(DOWNLOAD_DIR) / 'ADEChallengeData2016.zip'
    if download_path.exists():
        print(f'Download path {download_path} already exists! download step is skipped.')
    else:
        torch.hub.download_url_to_file(ADE20K_URL, download_path)

    # Set base directory
    ade20k_path = Path(args.dir) / 'ade20k'
    os.makedirs(ade20k_path, exist_ok=True)

    # Extract images
    # print('Extracting images zip file... This may take a minutes.')
    tmp2extract = ade20k_path / 'tmp2extract'
    shutil.unpack_archive(download_path, tmp2extract, "zip")

    # Move train samples
    src_root = tmp2extract / 'ADEChallengeData2016'

    os.rename(src_root / 'images' / 'training', src_root / 'images' / 'train') # images/training -> images/train
    os.rename(src_root / 'images' / 'validation', src_root / 'images' / 'valid') # images/validation -> images/valid

    os.rename(src_root / 'annotations', src_root / 'labels') # annotations -> labels
    os.rename(src_root / 'labels' / 'training', src_root / 'labels' / 'train') # labels/training -> labels/train
    os.rename(src_root / 'labels' / 'validation', src_root / 'labels' / 'valid') # labels/validation -> labels/valid

    shutil.move(src_root / 'images', ade20k_path / 'images')
    shutil.move(src_root / 'labels', ade20k_path / 'labels')

    for phase in ['train', 'valid']:
        phase_path = ade20k_path / 'labels' / phase
        labels = os.listdir(phase_path)
        for label_name in labels:
            label = Image.open(phase_path / label_name)
            label = np.array(label)
            label[label == 0] = 255 # Ignore backgrounds
            label[label != 255] -= 1 # Shift label values
            label = Image.fromarray(label)
            label.save(phase_path / label_name)

    # Build id_mapping.json
    with open(ade20k_path / 'id_mapping.json', 'w') as f:
        json.dump(ID_MAPPING, f)

    try:
        shutil.rmtree(tmp2extract)
    except OSError as e:
        print(e)
