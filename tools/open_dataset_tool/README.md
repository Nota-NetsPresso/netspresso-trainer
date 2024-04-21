## Dataset preparation (Local)

If you are interested in utilizing open datasets, you can exploit them by following the instructions.

### Image classification

<details>
<summary>CIFAR100</summary>
 
Simply run `cifar100.py` python file with your dataset directory as an argument.

CIFAR100 dataset will be automatically downloaded to `./data/download`. After executing scripts, you can use  [pre-defined configuration](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/config/data/cifar100.yaml).

```bash
python ./tools/open_dataset_tool/cifar100.py --dir ./data
```
</details>

<details>
<summary>ImageNet1K</summary>

ImageNet1K dataset cannot be automatically downloaded. You should download dataset from [ImageNet](https://www.image-net.org/) website, and place downloaded files into `./data/download`.

And, run `imagenet1k.py` python file with your dataset directorty and downloaded files path as arguments. After executing scripts, you can use [pre-defined configuration]().

*(`imagenet1k.py` needs scipy library which is in [requirements-optional.txt](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/requirements-optional.txt))*

```bash
python ./tools/open_dataset_tool/imagenet1k.py --dir ./data --train-images ./data/download/ILSVRC2012_img_train.tar --valid-images ./data/download/ILSVRC2012_img_val.tar --devkit ./data/download/ILSVRC2012_devkit_t12.tar.gz
```
</details>

### Semantic segmentation

<details>
<summary>PascalVOC 2012</summary>
 
Simply run `voc2012_seg.py` python file with your dataset directory as an argument.

PascalVOC 2012 dataset will be automatically downloaded to `./data/download`. After executing scripts, you can use  [pre-defined configuration](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/config/data/voc12.yaml).

```bash
python ./tools/open_dataset_tool/voc2012_seg.py --dir ./data
```
</details>

### Object detection

<details>
<summary>COCO 2017</summary>

Simply run `coco2017.py` python file with your dataset directory as an argument.

COCO 2017 dataset will be automatically downloaded to `./data/download`. After executing scripts, you can use  [pre-defined configuration](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/config/data/coco2017.yaml).

```bash
python ./tools/open_dataset_tool/coco2017.py --dir ./data
```
</details>
