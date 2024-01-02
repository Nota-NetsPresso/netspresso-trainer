# Overview

NetsPresso Trainer and NetsPresso service provides a convenient experience in training, compressing, retraining, and deploying models for user devices, seamlessly. In that process, NetsPresso Trainer manages both training and retraining phases, ensuring the models are fully compatible with NetsPresso.

NetsPresso Trainer categorizes essential parameters for training into six configuration modules. Each module is responsible for the following aspects:

- **Data**: Defines the structure of the user-customized or Hugging Face datasets for NetsPresso Trainer to read.
- **Augmentation**: Defines the data augmentation recipe.
- **Model**: Defines the model configuration, loss modules, and pretrained weights.
- **Training**: Defines necessary elements like optimizer, epochs, and batch size for training.
- **Logging**: Defines output formats of training results.
- **Environment**: Defines the training environment, including GPU usage and dataloader 

This component section describes the six configuration modules (data, augmentation, model, training, logging, and environment) which are necessary to use NetsPresso Trainer. [You can see yaml configuration examples in our public repository.](https://github.com/Nota-NetsPresso/netspresso-trainer/tree/dev/config)

## Advantage of NetsPresso Trainer

### SOTA models fully compatible with NetsPresso

NetsPresso Trainer provides reimplemented SOTA models to ensure compatibility with NetsPresso. This allows users to avoid expending resources on changing model formats for model compression and device deployment, enabling the application of SOTA models to user-specific tasks.

### Easily trainable with yaml configuration

NetsPresso Trainer encapsulated all values in configurations which are required for model training. This enables extensive optimization attempts with mere modifications of these configuration files. Also, this enhances usability by using same configuration format for retraining compressed models.
