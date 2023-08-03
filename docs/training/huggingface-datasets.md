## Training with HuggingFace datasets

We do our best to give you a good experience in training process. We integrate [HuggingFace(HF) datasets](https://huggingface.co/datasets) into our training pipeline. Note that we apply our custom augmentation methods in training datasets, instead of albumentations which is mostly used in HF datasets.

To do so, firstly you need to install additional libraries with the following command:

```bash
pip install -r requirements-data.txt
```

Then, you can write your own data configuration for HF datasets. Please refer to [data configuration template](./config/data/template).  
Some datasets in HF datasets needs `login`. You can login with `huggingface-cli login` with their [official guide](https://huggingface.co/docs/huggingface_hub/quick-start#login).