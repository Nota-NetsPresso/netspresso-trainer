# Upcoming Release

## New Features:

- Construct head by config file by `@illian01` in [PR 237](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/237)
- Construct neck by config file by `@illian01` in [PR 249](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/249)
- Add model: RetinaNet by `@illian01` in [PR 257](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/257)
- Select `gpus` with `environment` configuration by `@deepkyu` in [PR 269](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/269)
- Return logging directory path and fix training interfaces by `@deepkyu` in [PR 271](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/271)

## Bug Fixes:

- Fix attribute error on fc by `@illian01` in [PR 252](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/252)
- Restore file export for stream log by `@deepkyu` in [PR 255](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/255)
- Fix CSV logging, configuration error, and misused loggings by `@deepkyu` in [PR 259](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/259)
- Fix minor bug in train.py by `@illian01` in [PR 277](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/277)

## Breaking Changes:

- Provide pytorch state dict with `.safetensors` and training summary with `.json` for a better utilization by `@deepkyu` in [PR 262](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/262)

## Other Changes:

- Refactoring for detection models by `@illian01` in [PR 260](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/260)
- Equalize logging format with `PyNetsPresso` by `@deepkyu` in [PR 263](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/263)
- Refactoring for clean docs by `@illian01` in [PR 265](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/265), [PR 266](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/266), [PR 272](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/272), [PR 273](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/273), [PR 274](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/274)
- Refactoring model building code and move TASK_MODEL_DICT by `@illian01` in [PR 282](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/282)

# v0.0.10

## New Features:

- Add a gpu option in `train_with_config` (only single-GPU supported) by `@deepkyu` in [PR 219](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/219)
- Support augmentation for classification task: cutmix, mixup by `@illian01` in [PR 221](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/221)
- Add model: MixNet by `@illian01` in [PR 229](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/229)
- Add `model.name` to get the exact nickname of the model by `@deepkyu` in [PR 243](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/243/)
- Add transforms: RandomErasing and TrivialAugmentationWide by `@illian01` in [PR 246](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/246)

## Bug Fixes:

- Fix PIDNet model dataclass task field by `@illian01` in [PR 220](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/220)
- Fix default criterion value of classification `@illian01` in [PR 238](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/238)
- Fix model access of 2-stage detection pipeline to compat with distributed environment by `@illian` in [PR 239](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/239)

## Breaking Changes:

- Enable dataset augmentation customizing by `@illian01` in [PR 201](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/201)
- Add postprocessor module by `@illian01` in [PR 223](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/223)
- Equalize the model backbone configuration format by `@illian01` in [PR 228](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/228)
- Separate FPN and PAFPN as neck module by `@illian01` in [PR 234](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/234)
- Auto-download pretrained checkpoint from AWS S3 by `@deepkyu` in [PR 244](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/244)

## Other Changes:

- Update ruff rule (`W`) by `@deepkyu` in [PR 218](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/218)
- Integrate classification loss modules by `@illian01` in [PR 226](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/226)

# v0.0.9

## New Features:

- Add YOLOX model by `@illian01` in [PR 195](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/195), [PR 212](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/212)
- Fix Faster R-CNN detection head to compat with PyNP compressor by `@illian01` in [PR 184](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/184), [PR 194](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/194), [PR 204](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/204)
- Support multi-GPU training with `netspresso-train` entrypoint by `@deepkyu`, `@illian01` and `@Only-bottle` in [PR 213](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/213)

## Bug Fixes:

- Remove fx training flag in entry point by `@illian01` in [PR 188](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/188)
- Fix bounding box coordinates computing error on random flip augmentation by `@illian01` in [PR 211](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/211)

## Breaking Changes:

- Release NetsPresso Trainer colab tutorial `@illian01` in [PR 191](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/191)
- Support training with python-level config by `@deepkyu` in [PR 205](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/205)

## Other Changes:

- Refactoring models/op module by `@illian01` in [PR 189](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/189), [PR 190](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/190)
- Parameterize activation function of BasicBlock and Bottleneck by `@illian01` in [PR193](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/193)
- Modify MobileNetV3 to stage format and remove forward hook by `@illian01` in [PR 199](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/199)
- Substitute MACs counter with `fvcore` library to sync with NetsPresso by `@deepkyu` and `@Only-bottle` in [PR 202](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/202)
- Enable to compute metric with all training samples by `@illian01` in [PR 210](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/210)

# v0.0.8

## New Features:

- Add MobileNetV3 by `@illian01` in [PR 173](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/173)
- Handling for fp16 fx model by `@illian01` in [PR 175](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/175)
- Deploy Gradio simulators to Hugging Face Space by `@deepkyu` in [PR 181](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/181)

## Bug Fixes:

No changes to highlight.

## Breaking Changes:

No changes to highlight.

## Other Changes:

- Removed `model_name` check in `create_transform_segmentation` function by `@illian01` in [Pr 176](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/176)
- Combine entry point to `train.py` by `@illian01` in [Pr 180](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/180)

# v0.0.7

## New Features:

No changes to highlight.

## Bug Fixes:

- ⚠️ Fix pypi package import by `@deepkyu` in [PR 169](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/169)

## Breaking Changes:

No changes to highlight.

## Other Changes:

No changes to highlight.


# v0.0.6

## New Features:

- Support RGB segmentation map and class with label value by `@deepkyu` in [PR 163](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/163)

## Bug Fixes:

- Fix import error for `Sequence` by `@illian01` in [PR 155](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/155)
- Add last epoch validation and delete save_converted_model by `@illian01` in [PR 157](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/157)

## Breaking Changes:

No changes to highlight.

## Other Changes:

- Add onnx save in best model saving step of graph model training by `@illian01` in [PR 160](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/160).
- Update to keep community standards by `@illian01` in [PR 162](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/162)
- Update a lot of contents in docs (but not finished...) by `@deepkyu` in [PR 165](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/165)
- Add github workflow for pypi packaging by `@deepkyu` in [PR 166](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/166)


# v0.0.5

Notice: there are some changes in maintaining the repository and we transferred the original `private` repository to the public(planned) location. Some PR links may be expired because those links are based on the previous version of repository. Hope you understand.  
This change is applied at [PR 151](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/151)

## New Features:

- Update the model configuration to handle the architecture by `@deepkyu` in [PR 130](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/130)

## Bug Fixes:

- Add `tensorboard` in requirements.txt by `@illian01` in [PR 134](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/134)
- Fix typo in `scripts/example_train.sh` by `@illian01` in [PR 137](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/137)
- Initialize loss and metric at same time with optimizer and lr schedulers by `@deepkyu` in [PR 138](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/138)
- Hotfix the error which shows 0 for validation loss and metrics by fixing the variable name by `@deepkyu` in [PR 140](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/140)
- Add missing field, `save_optimizer_state`, in `logging.yaml` by `@illian01` in [PR 149](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/149)
- Hotfix for pythonic config name (classification loss) by `@deepkyu` in [PR 242](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/242)

## Breaking Changes:

- Add checkpoint saving while training, resume training with the checkpoint, save the training summary with # Params and MACs by `@deepkyu` in [PR 135](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/135)
- Change parsing argument for FX model retraining and resuming training to model configuration by `@deepkyu` in [PR 135](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/135)
- Apply ruff linter and add workflow for ruff checking by `@deepkyu` in [PR 143](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/143)

## Other Changes:

- Add PyNetsPresso tab in documentation page by `@deepkyu` in [PR 128](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/128)
- Fix issue template and default assignee per issue type by `@deepkyu` in [PR 144](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/144)


# v0.0.4

## New Features:

- Generalize segmentation head and add support ResNet50 + segmentation by `@deepkyu` in [PR 122](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/122)

## Bug Fixes:

No changes to highlight.

## Breaking Changes:

No changes to highlight.

## Other Changes:

- Simplify training configuration and example training scripts by `@deepkyu` in [PR 124](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/124)
- Add `PyNetsPresso` tab in docs page by `@deepkyu` in [PR 128](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/128)


# v0.0.3

## New Features:

- Add LR simulator (powered by gradio) with `training` configuration by `@deepkyu` in `[PR 116](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/116)`
- Add augmentation simulator (powered by gradio) with `augmentation` configuration by `@deepkyu` in `[PR 118](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/118)`
- Add LR scheduler (cosine with warm restart, step_lr) by `@deepkyu` in `[PR 114](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/114)`

## Bug Fixes:

No changes to highlight.

## Breaking Changes:

- Support detection training with its metric by `@deepkyu` in `[PR 119](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/119)`

## Other Changes:

No changes to highlight.
