# Overview

NetsPresso Trainer provides data augmentation function to improve model performance, allowing users to configure their own training recipes as desired. 
Data augmentation in NetsPresso Trainer is based on torch and torchvision, and all augmentations are written based on `pillow` images and `torch.Tensor`.  

Currently, there is no function for users to directly implement augmentation recipes, but only customize variable values to adjust the intensity and frequency of augmentations. In the near future, a function to directly design augmentation recipes will be added. 

## Supporting transforms

- The currently supported augmentation methods in NetsPresso Trainer are as follows.

### ColorJitter

### Identity

### Normalize

### Pad

### PadIfNeeded

### RandomCrop

### RandomHorizontalFlip

### RandomResizedCrop

### RandomVerticalFlip

### Resize

### ToTensor

## Gradio demo for simulating the transform

In many learning function repositories, it is recommended to check the code and documentation for augmentations or actually run the training to check the logs to see how augmentations are performed. 
NetsPresso Trainer supports augmentation simulation to help users easily understand the augmentation recipe they have configured. 
By copying and entering the augmentation configuration into the simulator, users can preview how a specific image will be augmented in advance. However, transforms (e.g. Normalize, ToTensor) used to convert the image array for learning purposes are excluded from the simulation visualization process. 
In particular, since this simulator directly imports the augmentation modules used in NetsPresso Trainer, users can use the same functions as the augmentation functions used in actual training to verify the results.  

Our team hopes that the learning process with NetsPresso Trainer will become a more enjoyable experience for all users. 

### How to use

#### Running on your environment
*FIXME*

#### Hugging Face Spaces
*FIXME*

<script
	type="module"
	src="https://gradio.s3-us-west-2.amazonaws.com/3.28.0/gradio.js"
></script>

<gradio-app src="https://nota-ai-compressed-wav2lip.hf.space"></gradio-app>

## Field list