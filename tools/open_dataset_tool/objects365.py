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

import os
import time
import json
import torch
import shutil
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

DEFAULT_DATA_DIR = './data'
DOWNLOAD_DIR = './data/download/objects365'
CLASS365_NAME_TO_LABEL = {
    "Person": 0, "Sneakers": 1, "Chair": 2, "Other Shoes": 3, "Hat": 4, "Car": 5, "Lamp": 6, "Glasses": 7, "Bottle": 8, "Desk": 9,
    "Cup": 10, "Street Lights": 11, "Cabinet/shelf": 12, "Handbag/Satchel": 13, "Bracelet": 14, "Plate": 15, "Picture/Frame": 16, "Helmet": 17, "Book": 18, "Gloves": 19,
    "Storage box": 20, "Boat": 21, "Leather Shoes": 22, "Flower": 23, "Bench": 24, "Potted Plant": 25, "Bowl/Basin": 26, "Flag": 27, "Pillow": 28, "Boots": 29,
    "Vase": 30, "Microphone": 31, "Necklace": 32, "Ring": 33, "SUV": 34, "Wine Glass": 35, "Belt": 36, "Monitor/TV": 37, "Backpack": 38, "Umbrella": 39,
    "Traffic Light": 40, "Speaker": 41, "Watch": 42, "Tie": 43, "Trash bin Can": 44, "Slippers": 45, "Bicycle": 46, "Stool": 47, "Barrel/bucket": 48, "Van": 49,
    "Couch": 50, "Sandals": 51, "Basket": 52, "Drum": 53, "Pen/Pencil": 54, "Bus": 55, "Wild Bird": 56, "High Heels": 57, "Motorcycle": 58, "Guitar": 59,
    "Carpet": 60, "Cell Phone": 61, "Bread": 62, "Camera": 63, "Canned": 64, "Truck": 65, "Traffic cone": 66, "Cymbal": 67, "Lifesaver": 68, "Towel": 69,
    "Stuffed Toy": 70, "Candle": 71, "Sailboat": 72, "Laptop": 73, "Awning": 74, "Bed": 75, "Faucet": 76, "Tent": 77, "Horse": 78, "Mirror": 79,
    "Power outlet": 80, "Sink": 81, "Apple": 82, "Air Conditioner": 83, "Knife": 84, "Hockey Stick": 85, "Paddle": 86, "Pickup Truck": 87, "Fork": 88, "Traffic Sign": 89,
    "Balloon": 90, "Tripod": 91, "Dog": 92, "Spoon": 93, "Clock": 94, "Pot": 95, "Cow": 96, "Cake": 97, "Dinning Table": 98, "Sheep": 99,
    "Hanger": 100, "Blackboard/Whiteboard": 101, "Napkin": 102, "Other Fish": 103, "Orange/Tangerine": 104, "Toiletry": 105, "Keyboard": 106, "Tomato": 107, "Lantern": 108, "Machinery Vehicle": 109,
    "Fan": 110, "Green Vegetables": 111, "Banana": 112, "Baseball Glove": 113, "Airplane": 114, "Mouse": 115, "Train": 116, "Pumpkin": 117, "Soccer": 118, "Skiboard": 119,
    "Luggage": 120, "Nightstand": 121, "Tea pot": 122, "Telephone": 123, "Trolley": 124, "Head Phone": 125, "Sports Car": 126, "Stop Sign": 127, "Dessert": 128, "Scooter": 129,
    "Stroller": 130, "Crane": 131, "Remote": 132, "Refrigerator": 133, "Oven": 134, "Lemon": 135, "Duck": 136, "Baseball Bat": 137, "Surveillance Camera": 138, "Cat": 139,
    "Jug": 140, "Broccoli": 141, "Piano": 142, "Pizza": 143, "Elephant": 144, "Skateboard": 145, "Surfboard": 146, "Gun": 147, "Skating and Skiing shoes": 148, "Gas stove": 149,
    "Donut": 150, "Bow Tie": 151, "Carrot": 152, "Toilet": 153, "Kite": 154, "Strawberry": 155, "Other Balls": 156, "Shovel": 157, "Pepper": 158, "Computer Box": 159,
    "Toilet Paper": 160, "Cleaning Products": 161, "Chopsticks": 162, "Microwave": 163, "Pigeon": 164, "Baseball": 165, "Cutting/chopping Board": 166, "Coffee Table": 167, "Side Table": 168, "Scissors": 169,
    "Marker": 170, "Pie": 171, "Ladder": 172, "Snowboard": 173, "Cookies": 174, "Radiator": 175, "Fire Hydrant": 176, "Basketball": 177, "Zebra": 178, "Grape": 179,
    "Giraffe": 180, "Potato": 181, "Sausage": 182, "Tricycle": 183, "Violin": 184, "Egg": 185, "Fire Extinguisher": 186, "Candy": 187, "Fire Truck": 188, "Billiards": 189,
    "Converter": 190, "Bathtub": 191, "Wheelchair": 192, "Golf Club": 193, "Briefcase": 194, "Cucumber": 195, "Cigar/Cigarette": 196, "Paint Brush": 197, "Pear": 198, "Heavy Truck": 199,
    "Hamburger": 200, "Extractor": 201, "Extension Cord": 202, "Tong": 203, "Tennis Racket": 204, "Folder": 205, "American Football": 206, "earphone": 207, "Mask": 208, "Kettle": 209,
    "Tennis": 210, "Ship": 211, "Swing": 212, "Coffee Machine": 213, "Slide": 214, "Carriage": 215, "Onion": 216, "Green beans": 217, "Projector": 218, "Frisbee": 219,
    "Washing Machine/Drying Machine": 220, "Chicken": 221, "Printer": 222, "Watermelon": 223, "Saxophone": 224, "Tissue": 225, "Toothbrush": 226, "Ice cream": 227, "Hot-air balloon": 228, "Cello": 229,
    "French Fries": 230, "Scale": 231, "Trophy": 232, "Cabbage": 233, "Hot dog": 234, "Blender": 235, "Peach": 236, "Rice": 237, "Wallet/Purse": 238, "Volleyball": 239,
    "Deer": 240, "Goose": 241, "Tape": 242, "Tablet": 243, "Cosmetics": 244, "Trumpet": 245, "Pineapple": 246, "Golf Ball": 247, "Ambulance": 248, "Parking meter": 249,
    "Mango": 250, "Key": 251, "Hurdle": 252, "Fishing Rod": 253, "Medal": 254, "Flute": 255, "Brush": 256, "Penguin": 257, "Megaphone": 258, "Corn": 259,
    "Lettuce": 260, "Garlic": 261, "Swan": 262, "Helicopter": 263, "Green Onion": 264, "Sandwich": 265, "Nuts": 266, "Speed Limit Sign": 267, "Induction Cooker": 268, "Broom": 269,
    "Trombone": 270, "Plum": 271, "Rickshaw": 272, "Goldfish": 273, "Kiwi fruit": 274, "Router/modem": 275, "Poker Card": 276, "Toaster": 277, "Shrimp": 278, "Sushi": 279,
    "Cheese": 280, "Notepaper": 281, "Cherry": 282, "Pliers": 283, "CD": 284, "Pasta": 285, "Hammer": 286, "Cue": 287, "Avocado": 288, "Hamimelon": 289,
    "Flask": 290, "Mushroom": 291, "Screwdriver": 292, "Soap": 293, "Recorder": 294, "Bear": 295, "Eggplant": 296, "Board Eraser": 297, "Coconut": 298, "Tape Measure/Ruler": 299,
    "Pig": 300, "Showerhead": 301, "Globe": 302, "Chips": 303, "Steak": 304, "Crosswalk Sign": 305, "Stapler": 306, "Camel": 307, "Formula 1": 308, "Pomegranate": 309,
    "Dishwasher": 310, "Crab": 311, "Hoverboard": 312, "Meat ball": 313, "Rice Cooker": 314, "Tuba": 315, "Calculator": 316, "Papaya": 317, "Antelope": 318, "Parrot": 319,
    "Seal": 320, "Butterfly": 321, "Dumbbell": 322, "Donkey": 323, "Lion": 324, "Urinal": 325, "Dolphin": 326, "Electric Drill": 327, "Hair Dryer": 328, "Egg tart": 329,
    "Jellyfish": 330, "Treadmill": 331, "Lighter": 332, "Grapefruit": 333, "Game board": 334, "Mop": 335, "Radish": 336, "Baozi": 337, "Target": 338, "French": 339,
    "Spring Rolls": 340, "Monkey": 341, "Rabbit": 342, "Pencil Case": 343, "Yak": 344, "Red Cabbage": 345, "Binoculars": 346, "Asparagus": 347, "Barbell": 348, "Scallop": 349,
    "Noddles": 350, "Comb": 351, "Dumpling": 352, "Oyster": 353, "Table Tennis paddle": 354, "Cosmetics Brush/Eyeliner Pencil": 355, "Chainsaw": 356, "Eraser": 357, "Lobster": 358, "Durian": 359,
    "Okra": 360, "Lipstick": 361, "Cosmetics Mirror": 362, "Curling": 363, "Table Tennis": 364
}
CLASS365_LABEL_TO_NAME = {val: key for key, val in CLASS365_NAME_TO_LABEL.items()}

def txtywh2cxcywh(top_left_x, top_left_y, width, height):
    cx = top_left_x + (width / 2)
    cy = top_left_y + (height / 2)
    w = width
    h = height
    return cx, cy, w, h


def cxcywh2cxcywhn(cx, cy, w, h, img_w, img_h):
    return cx / img_w, cy / img_h, w / img_w, h / img_h

def download_file(url, path, max_retries=30, delay=5, silent=True):
    silent_option = "sS" if silent else ""
    for attempt in range(max_retries):
        try:
            subprocess.run([
                "curl", "-#", f"-{silent_option}L", "-o", str(path),
                "--connect-timeout", "30",
                "--max-time", "300",
                "-C", "-",
                url
            ], check=True)
            print(f"Successfully downloaded {url} to {path}")
            return
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt + 1} failed to download {url}: {e}")
            print(f"Retry download after {delay} seconds ...")
            if attempt < max_retries - 1:
                time.sleep(delay)
    raise Exception(f"Failed to download {url} after {max_retries} attempts")

def download_worker(i, split, base_url, images_dir, silent=True):
    print(f'Download {split} images patch {i}...')
    image_download_path = Path(DOWNLOAD_DIR) / f'objects365_{split}_patch{i}.tar.gz'
    version = 1 if i < 16 else 2
    download_url = f'{base_url}images/v{version}/patch{i}.tar.gz' if split == 'val' else f'{base_url}patch{i}.tar.gz'
    download_file(download_url, image_download_path, silent=silent)

    print(f'Unzip {split} images {image_download_path} file ...')
    subprocess.run(["tar", "xfz", image_download_path, "--directory", images_dir, "--strip-components", '1'], check=True)
    print(f'Done patch {i}')
        

def process_annotations(annotation_path, label_dir):
    with open(annotation_path) as f:
        ann_json = json.load(f)

    annotations = {image_info['id']: [image_info['file_name']] for image_info in ann_json['images']}
    imgid_to_info = {info['id']: info for info in ann_json['images']}

    for ann in tqdm(ann_json['annotations']):
        image_id = ann['image_id']
        label = ann['category_id'] - 1
        top_left_x, top_left_y, width, height = ann['bbox']
        cx, cy, w, h = txtywh2cxcywh(top_left_x, top_left_y, width, height)
        cx, cy, w, h = cxcywh2cxcywhn(cx, cy, w, h, imgid_to_info[image_id]['width'], imgid_to_info[image_id]['height'])
        instance = [label, cx, cy, w, h]
        annotations[image_id].append(instance)

    for _, info in tqdm(annotations.items()):
        file_name = info[0].split('/')[-1]
        texts = '\n'.join([f'{" ".join(map(str, line))}' for line in info[1:]])
        with open((label_dir / file_name).with_suffix('.txt'), 'w') as f:
            f.write(texts)

def process_split(split, patches, base_url, objects365_path, num_process):
    ann_ext = '.tar.gz' if split == 'train' else '.json'
    ann_download_path = Path(DOWNLOAD_DIR) / f'objects365_annotation_{split}{ann_ext}'
    annotation_dir = objects365_path / 'annotations'
    annotation_path = annotation_dir / f'zhiyuan_objv2_{split}.json'
    label_dir = objects365_path / 'labels' / split
    images_dir = objects365_path / 'images' / split

    for dir_path in [label_dir, images_dir, annotation_dir]:
        shutil.rmtree(dir_path, ignore_errors=True)
        os.makedirs(dir_path, exist_ok=True)

    if not ann_download_path.exists():
        print(f'Download {split} annotation file ...')
        download_url = f'{base_url}zhiyuan_objv2_{split}'
        torch.hub.download_url_to_file(f'{download_url}{ann_ext}', ann_download_path)

    if split == 'train':
        print('Unzip training annotation tar.gz file ...')
        shutil.unpack_archive(ann_download_path, annotation_dir, "gztar")
    else:
        print('Moving validation annotation .json file to the appropriate location ...')
        shutil.copyfile(ann_download_path, annotation_path)

    args = [(i, split, base_url, images_dir, num_process > 1) for i in range(patches)]
    with Pool(num_process) as pool:
        pool.starmap(download_worker, args)

    print(f"All {split} image patches processed")
    print(f'Building {split} labels ...')

    process_annotations(annotation_path, label_dir)

def main():
    parser = argparse.ArgumentParser(description="Parser for objects365 dataset downloader.")
    parser.add_argument('--dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--num_process', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    objects365_path = Path(args.dir) / 'objects365'
    os.makedirs(objects365_path, exist_ok=True)

    splits = {
        'train': {'patches': 51, 'base_url': "https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/"},
        'val': {'patches': 44, 'base_url': "https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/"}
    }

    for split, info in splits.items():
        process_split(split, info['patches'], info['base_url'], objects365_path, args.num_process)

    id_mapping = [CLASS365_LABEL_TO_NAME[i] for i in range(365)]
    with open(objects365_path / 'id_mapping.json', 'w') as f:
        json.dump(id_mapping, f)

    shutil.rmtree(objects365_path / 'annotations', ignore_errors=True)

if __name__ == "__main__":
    main()