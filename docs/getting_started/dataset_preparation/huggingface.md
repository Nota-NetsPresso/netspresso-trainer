# Data preparation (Hugging Face)

The Hugging Face datasets offers a vast array of datasets to support various tasks, making them readily accessible through a user-friendly API. Provided datasets by Hugging Face datasets are typically structured into `training`, `validation`, and `testing` sets. This structure allows NetsPresso Trainer to utilize various datasets with yaml configuration.

To explore official Hugging Face datasets catalogue, please refer to the [hugging Face datasets page](https://huggingface.co/datasets).

## Hugging Face datasets install

First, you must install the hugging Face datasets library.

```bash
pip install -r requirements-optional.txt

or

pip install datasets
```

## Find dataset repository

We use [CIFAR100](https://huggingface.co/datasets/cifar100) dataset as an example. Thus, `format` field in data configuration is filled as huggingface, and `metadata.repo` is filled as `cifar100`.

```yaml
data:
  name: cifar100
  task: classification
  format: huggingface
  metadata:
    custom_cache_dir: ./data/huggingface 
    repo: cifar100
    subset: ~
    features:
      image: ~
      label: ~
```

## Agreement the conditions to access the datasets

Some datasets are publicly available, but may require a agreement to be used. For instance, to use [ImageNet1K](https://huggingface.co/datasets/imagenet-1k) in Hugging Face datasets, you have to log in to Hugging Face homepage and accpet the conditions.

Make sure that you agreed to the conditions on Hugging Face website, and log in to `huggingface-cli` before you start.

```bash
huggingface-cli login
```

## Set subset

Some datasets have multiple subsets in their dataset hierarchy. They have to be specified in `subset` field.

If there is no subset in the dataset, you can leave `subset` field as null.

```yaml
data:
  name: cifar100
  task: classification
  format: huggingface
  metadata:
    custom_cache_dir: ./data/huggingface 
    repo: cifar100
    subset: ~ # You have to fill this field if there is subset in the dataset you trying to use
    features:
      image: ~
      label: ~
```

## Set features

You should check features of dataset in Hugging Face homepage. If you see [CIFAR100](https://huggingface.co/datasets/cifar100), there are three features in the dataset which are `img`, `fine_label`, `coarse_label`. In this dataset, the image data is denoted by `img` and the labels for the 100 classes are represented by `fine_label`. Given this structure, the data configuration should be filled out as below.

```yaml
data:
  name: cifar100
  task: classification
  format: huggingface
  metadata:
    custom_cache_dir: ./data/huggingface 
    repo: cifar100
    subset: ~ # You should fill this field if there is subset in the dataset you trying to use
    features:
      image: img
      label: fine_label
```

## Run NetsPresso Trainer

Now you can run NetsPresso Trainer with Hugging Face dataset!

```bash
python train.py --data your_huggingface_dataset_yaml_path.yaml ...
```
