# Augmentation - Overview

NetsPresso Trainer provides data augmentation function to improve model performance, allowing users to configure their own training recipes as desired. 
Data augmentation in NetsPresso Trainer is based on torch and torchvision, and all augmentations are written based on `pillow` images and `torch.Tensor`.  

In NetsPresso Trainer, users can create their desired augmentation recipe by composing a configuration as below. We categorizd augmentation functions into transforms and mix_transforms. Transforms contains methods that applied to each individual image, while mix_transforms include techniques that applied by mixing samples together.

Functions specified in the `transforms` field are applied sequentially as listed, while for `mix_transforms`, a single function is randomly chosen and applied to each data batch. Note that after all image processing is completed, the final image size must match with the size specified in `augmentation.img_size`.


```yaml
augmentation:
  img_size: &img_size 256
  transforms:
    - 
      name: randomresizedcrop
      size: *img_size
      interpolation: bilinear
    - 
      name: randomhorizontalflip
      p: 0.5
  mix_transforms:
    -
      name: cutmix
      alpha: 0.01
```

## Gradio demo for simulating the transform

In many learning function repositories, it is recommended to check the code and documentation for augmentations or actually run the training to check the logs to see how augmentations are performed. 
NetsPresso Trainer supports augmentation simulation to help users easily understand the augmentation recipe they have configured. 
By copying and entering the augmentation configuration into the simulator, users can preview how a specific image will be augmented in advance. However, transforms (e.g. Normalize, ToTensor) used to convert the image array for learning purposes are excluded from the simulation visualization process. 
In particular, since this simulator directly imports the augmentation modules used in NetsPresso Trainer, users can use the same functions as the augmentation functions used in actual training to verify the results.  

Our team hopes that the learning process with NetsPresso Trainer will become a more enjoyable experience for all users. 

### Running on your environment

Please run the gradio demo with following command:

```
bash scripts/run_simulator_augmentation.sh
```

## Field list

| Field <img width=200/> | Description |
|---|---|
| `augmentation.img_size` | (int) the image size of model input after finishing the data augmentation |
| `augmentation.transforms` | (list[dict]) List of transform functions. Augmentation process is defined on list order. |
| `augmentation.mix_trasforms` | (list[dict]) List of mix_transform functions. Mix transforms are applied after transform functions have been executed, and if multiple mix_transforms are listed, only one mix transform is applied per batch. |


