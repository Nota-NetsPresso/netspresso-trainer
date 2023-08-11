
## Models

### Features

- Compatible with torch.fx converting
- Can be compressed with pruning method provided in [NetsPresso](https://netspresso.ai)
- Efficient to be easily deployed at many edge devices.

### ViT and MetaFormer

![MetaFormer](https://user-images.githubusercontent.com/49296856/177275244-13412754-3d49-43ef-a8bd-17c0874c02c1.png)

[MetaFormer](https://arxiv.org/abs/2210.13452), introduced by [Sea AI Lab](https://sail.sea.com/), suggested the abstracted architecture of Vision Transformer and has widely applied in recent vision backbone architectures with better performance. The framework of MetaFormer not just covers the original ViT based models, but also explains some hybrid models of ConvNets and ViT in single framework. Especially, those models set new record in Efficient vision task, which show better performance in both accuracy and inference speed.  

From the MetaFormer paper, the concept of MetaFormer can be expressed as follows:

$$
X = \mathrm{InputEmbedding}(I), \newline
X' = \mathrm{MetaFormerEncoder}(X), \newline
X'' = \mathrm{NormOrIdentity_{output}}(X'),
$$

where $\mathrm{MetaFormerEncoder}$ usually consists of repeated MetaFormer blocks.

One of MetaFormer blocks can be expressed as follows:

$$
X' = X + \mathrm{TokenMixer}\left(\mathrm{Norm_1}(X)\right), \newline
X'' = X' + \mathrm{ChannelMLP}\left(\mathrm{Norm_2}(X')\right),
$$
where $\mathrm{TokenMixer}$ could be either $\mathrm{MultiHeadSelfAttention}$, $\mathrm{Identity}$, or $\mathrm{ConvLayers}$, and $\mathrm{ChannelMLP}$ usually consists of `Linear` layers and activation function.

Inspired from the MetaFormer research, we try out best to design and build models (both ViT and non-ViT models) with MetaFormer framework to utilize in various ways. By defining models with unified MetaFormer block, we can apply to all models whenever NetsPresso services progressed. For example, Compressor API of [PyNetsPresso](https://py.netspresso.ai/) supports structural pruning for those models, which accelerates the inference only with negotiable performance drop.

### Retraining the model from NetsPresso

If you got compressed model from NetsPresso, then it's time to retrain your model to get the best performance.

