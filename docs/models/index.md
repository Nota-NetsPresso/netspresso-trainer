
## Models

### Features

- Compatible with torch.fx converting
- Can be compressed with pruning method provided in [NetsPresso](https://netspresso.ai)
- Efficient to be easily deployed at many edge devices.

### ViT and MetaFormer

![MetaFormer](https://user-images.githubusercontent.com/49296856/177275244-13412754-3d49-43ef-a8bd-17c0874c02c1.png)

[MetaFormer](https://arxiv.org/abs/2210.13452), introduced by [Sea AI Lab](https://sail.sea.com/), suggested the abstracted architecture of Vision Transformer and has widely applied in recent vision backbone architectures with better performance. The framework of MetaFormer not just covers the original ViT based models, but also explains some hybrid models of ConvNets and ViT in single framework. Especially, those models set new record in Efficient vision task, which show better performance in both accuracy and inference speed.  

From the MetaFormer paper, the concept of MetaFormer can be expressed as follows:



Inspired from this research, we try out best to design and build models with MetaFormer framework to utilize in various ways. By defining models with unified MetaFormer block, we can apply to all models whenever NetsPresso services progressed.

### Retraining the model from NetsPresso

You got compressed model from NetsPresso? Then, it's time to retrain your model to get the best performance!