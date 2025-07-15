# v1.4.1

## New Features:

- Add yolov9 pretrained weights by `@illian01` in [PR 631](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/631)
- Add EXIR exporting feature by `@illian01` in [PR 632](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/632)

## Bug Fixes:

No changes to highlight.

## Breaking Changes:

No changes to highlight.

## Other Changes:

Fix/add data params mlflow by `@hglee98` in [PR 629](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/629)

# 1.4.0

## New Features:

- Add num_save_samples option and code refactoring by `@illian01` in [PR 625](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/625)

## Bug Fixes:

- Fix classification postprocessing bug by `@illian01` in [PR 627](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/627)

## Breaking Changes:

No changes to highlight.

## Other Changes:

- Refactoring sample saving logic by `@illian01` in [PR 626](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/626)
- Fix dockerfile torch version by `hglee98` in [PR 624](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/624)

# v1.3.1

## New Features:

- Upload ONNX model to mlflow server if mlflow logging option is enabled by `hglee98` in [PR 621](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/621)

## Bug Fixes:

- Fix bug where training failed with torch.fx detection model by `hglee98` in [PR 619](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/619)

## Breaking Changes:

No changes to highlight.

## Other Changes:

- Loosen torch version coverage, not fixed version by `@illian01` in [PR 622](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/622)

# v1.3.0

## New Features:

- Add MLFlow train tracking feature by `@hglee98` in [PR 615](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/615)

## Bug Fixes:

No changes to highlight.

## Breaking Changes:

No changes to highlight.

## Other Changes:

No changes to highlight.

# v1.2.0

## New Features:

- Add cache_data option by `@hglee98` in [PR 608](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/608)

## Bug Fixes:

- Fix augmentation config of yolox training example by `@hglee98` in [PR 610](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/610)

## Breaking Changes:

- Upgrade torch to 2.0.1 by `@illian01` in [PR 613](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/613)

## Other Changes:

- Delete label_size, targets parameters from forward method across models by `@hglee98` in [PR 611](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/611)

# v1.1.0

## New Features:

- Add instance number logging feature for detection task by `@hglee98` in [PR 577](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/577)
- Add Precision and Recall metric for detection task by `@hglee98` in [PR 579](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/579)
- Add YOLOv9 by `@hglee98` in [PR 585](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/585), [PR 586](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/586), [PR 592](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/592), [PR 593](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/593), [PR 595](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/595), [PR 589](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/589), [PR 590](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/590), [PR 597](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/597), [PR 598](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/598), [PR 601](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/601), [PR 602](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/602)
- Add prediction values to evaluation summary by `@hglee98` in [PR 600](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/600)

## Bug Fixes:

- Fix error of checkpoint saving by `@hglee98` in [PR 580](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/580)
- Fix example scripts by `@hglee98` in [PR 596](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/596)

## Breaking Changes:

No changes to highlight.

## Other Changes:

- Refactor RT-DETR and generalize CSPRepLayer and RepVGG block by `@hglee98` in [PR 581](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/581), [PR 594](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/594)
- Generalize 2d pooling layers and define as custom layer by `@hglee98` in [PR 583](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/583)
- Unify bbox transformation and IoU computing methods by `@hglee98` in [PR 587](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/587)
- Update documents by `@hglee98` in [PR 591](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/591)
- Update mosaic augmentation epoch specification by `@hglee98` in [PR 604](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/604)

# v1.0.3

## New Features:

- Add RandomResize2 by `@illian01` in [PR 550](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/550)
- Add confidence score on detection visualization by `@hglee98` in [PR 552](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/552)
- Add save_best_only option for saving model by `@hglee98` in [PR 555](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/555), [PR 567](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/567)
- Add option to select best model saving criterion by `@hglee98` and `@illian01` in [PR 557](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/557), [PR 573](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/573), [PR 574](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/574)
- Add MultiStepLR scheduler by `@hglee98` in [PR 559](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/559)
- Add class-wise metric analysis option by `@illian01` in [PR 568](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/568)
- Add ReLU6 by `@hglee98` in [PR 566](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/566)
- Add TFLite model evaluation feature by `@hglee98` in [PR 563](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/563)
- Add YOLO-Fastest-v2 by `@hglee98` in [PR 548](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/548)
- Add tabulating step for metric standard outputs by `@illian01` in [PR 570](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/570)

## Bug Fixes:

- Fix keyword error of segmentation training by `@illian01` in [PR 551](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/551)
- Fix typo when saving optimizer state_dict by `@hglee98` in [PR 553](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/553)
- Fix not initialized save_dtype error by `@hglee98` in [PR 565](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/565)
- Fix mAP error in case of certain classes object is not in the dataset `@hglee98` in [PR 571](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/571), [PR 572](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/572)

## Breaking Changes:

- Massive refactoring metric modules and add flexible metric selecting option by `@illian01` in [PR 564](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/564)

## Other Changes:

No changes to highlight.

# v1.0.2

## New Features:

- Add no_weight_decay and overwrite option in optimizer config by `@illian01` in [PR 534](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/534)
- Fuse reparameterable layers before save by `@hglee98` in [PR 531](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/531)
- Add ONNX model evaluation feature by `@illian01` in [PR 539](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/539)
- Add ONNX export tool by `@illian01` in [PR 541](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/541)
- Add VOC 2012 dataset auto-downloader by `@illian01` in [PR 546](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/546)

## Bug Fixes:

- Fix bug in RandomCrop and segmentation dataset getitem by `@illian01` in [PR 535](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/535)
- Handle unexpected error during training rt-detr by `@hglee98` in [PR 530](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/530)
- Fix no 'deploy' attribute error in fx model train phase by `@illian01` in [PR 543](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/543)
- Fix bugs of PIDNet forward and saving by `@illian01` in [PR 544](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/544)

## Breaking Changes:

- Remove fx_model_path in model configuration and parsing model format by file extensions by `@illian01` in [PR 538](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/538)

## Other Changes:

- Document updates by `@illian01` in [PR 533](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/533)
- Delete install command for the optional requirements from Dockerfile by `@hglee98` in [PR 532](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/532)

# v1.0.1

## New Features:

- Add RT-DETR by `@illian01` and `@hglee98` in [PR 490](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/490), [PR 491](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/491), [PR 494](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/494), [PR 498](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/498#pullrequestreview-2234711140), [PR 500](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/500), [PR 507](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/507), [PR 501](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/501)
- Add Objects365 dataset auto downloader by `@hglee98` in [PR 482](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/482)
- Add ResNet model parameters to support various form by `@illian01` in [PR 497](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/497), [PR 523](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/523)
- Add RandomIoUCrop, RandomZoomOut augmentation by `@hglee98` in [PR 504](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/504)
- Add MobileNetV4 backbone by `@illian01` and `hglee98` in [PR 516](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/516), [PR 520](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/520), [PR 526](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/526)
- Add gradient clipping feature by `hglee98` in [PR 506](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/506#pullrequestreview-2262064548)
- Enabled to control `ToTensor` and `Normalization` through config file by `@illian01` in [PR 522](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/522)
- Enable to set onnx opset version through config file by `@illian01` in [PR 525](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/525)

## Bug Fixes:

- Fix handling error in case of error occured in first epoch by `@illian01` in [PR 493](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/493)
- Fix error in FLOPs computation by `@illian01` in [PR 499](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/499)

## Breaking Changes:

No changes to highlight.

## Other Changes:

- Update pi 4b deployment benchmark by `@illian01` in [PR 492](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/492)
- Clamp bbox for detection dataset loading by `@hglee98` in [PR 503](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/503#pullrequestreview-2247173939)
- Combine requirements and requirements-optional by `@illian01` in [PR 517](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/517)

# v1.0.0

## New Features:

- Add YOLOX-nano and YOLOX-tiny by `@hglee98` in [PR 467](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/467)
- Separate postprocessor configuration hierarchy by `@hglee98` in [PR 470](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/470)
- Add YOLO-Fastest by `@hglee98` in [PR 471](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/471)
- Write detailed status on training_summary by `@illian01` in [PR 487](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/487)

## Bug Fixes:

No changes to highlight.

## Breaking Changes:

No changes to highlight.

## Other Changes:

- Change attention bias interpolate method by `@illian01` in [PR 468](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/468)
- Fix ViT token number in positional encoding for torch.fx compile step by `@illian01` in [PR 475](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/475)
- Remove output typing of PIDNet by `@illian01` in [PR 477](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/477)
- Update documentation by `@illian01` in [PR 483](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/483), [PR 485](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/485)
- Minor refactorings by `hglee98` in [PR 513](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/513), [PR 518](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/518#pullrequestreview-2262154284)

# v0.2.2

## New Features:

- Update Benchmarks & Checkpoints (docs) and weights files to fully usable by `@illian01` in [PR 446](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/446), [PR 447](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/447), [PR 456](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/456), [PR 461](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/461)
- Add TFLite runtime code example by `@illian01` in [PR 449](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/449)

## Bug Fixes:

- Fix best_epoch init error in TrainingSummary in case of training resume by `@illian01` in [PR 448](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/448)
- Fix segmentation metric logic bug by `@illian01` in [PR 455](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/455), [PR 460](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/460)

## Breaking Changes:

No changes to highlight.

## Other Changes:

- Refactoring: remove thop, replace MACs with FLOPs by `@illian01` in [PR 444](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/444)
- Add copyright for entire project by `@illian01` in [PR 451](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/451)
- Modify ImageSaver to receive various resolution at once `@illian01` in [PR 458](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/458)

# v0.2.1

## New Features:

- Add dataset validation step and refactoring data modules by `@illian01` in [PR 417](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/417), [PR 419](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/419)
- Add various dataset examples including automatic open dataset format converter by `@illian01` in [PR 430](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/430)
- Allow using text file path for the `id_mapping` field by `@illian01` in [PR 432](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/432), [PR 435](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/435)

## Bug Fixes:

- Fix test directory check line by `@illian01` in [PR 428](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/428)
- Fix Dockerfile installation commandline `@cbpark-nota` in [PR 434](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/434)

## Breaking Changes:

No changes to highlight.

## Other Changes:

- Save training summary at every end of epochs by `@illian01` in [PR 420](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/420)
- Refacotring: rename postprocessors/register.py to registry.py by `@aychun` in [PR 424](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/424)
- Add example configuration set by `@illian01` in [PR 438](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/438)
- Documentation: fix simple use config file path by `@cbpark-nota` in [PR 437](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/437)

# v0.2.0

## New Features:

- Add activation and dropout layer in FC by `@illian01` in [PR 325](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/325), [PR 327](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/327)
- Add function to Resize: Match longer side with input size and keep ratio by `@illian01` in [PR 329](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/329)
- Add transforms: MosaicDetection by `@illian01` in [PR 331](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/331), [PR 337](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/337), [PR 397](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/397)
- Add transform: HSVJitter by `@illian01` in [PR 336](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/336), [PR 413](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/413)
- Add transforms: RandomResize by `@illian01` in [PR 341](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/341), [PR 344](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/344), [PR 398](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/398)
- Add model EMA (Exponential Moving Average) by `@illian01` in [PR 348](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/348)
- Add entry point for evaluation and inference by `@illian01` in [PR 374](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/374), [PR 379](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/379), [PR 381](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/381), [PR 383](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/383)
- Add classification visulizer by `@illian01` in [PR 384](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/384)
- Add dataset caching feature by `@illian01` in [PR 391](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/391)
- Add mixed precision training by `@illian01` in [PR 392](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/392)
- Add YOLOX l1 loss activation option by `@illian01` in [PR 396](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/396)
- Add NetsPresso Trainer YOLOX pretrained weights by `@illian01` in [PR 406](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/406)

## Bug Fixes:

- Fix output_root_dir from fixed string to config value by `@illian01` in [PR 323](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/323)
- Gather predicted results before compute metric and fix additional distributed evaluation inaccurate error by `@illian01` in [PR 346](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/346), [PR 356](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/356)
- Fix detection score return by `@illian01` in [PR 373](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/373)
- Fix memory leak from onnx export by `@illian01` in [PR 386](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/386), [PR 394](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/394)
- Refactoring metric modules and fix inaccurate metric bug by `@illian01` in [PR 402](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/402)

## Breaking Changes:

- Simplify augmentation configuration hierarchy by `@illian01` in [PR 322](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/322)
- Add pose estimation task and RTMPose model by `@illian01` in [PR 357](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/357), [PR 366](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/366)
- Remove pythonic config and move training initialization functions to `trainer_main.py` by `@illian01` in [PR 371](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/371)
- Unify gradio demo in one page by `@deepkyu` in [PR 408](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/408)

## Other Changes:

- Refactoring: Move custom transforms to each python module by `@illian01` in [PR 332](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/332)
- Update Pad transform to receive target size of image by `@illian01` in [PR 334](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/334)
- Rafactoring: Fix to make transform object in init by `@illian01` in [PR 339](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/339)
- ~~Add before_epoch step which does update modules like dataloader before epoch training by `@illian01` in [PR 340](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/340)~~
- Revert [PR 340](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/340) and add multiprocessing.Value to handle MosaicDetection and RandomResize by `@illian01` in [PR 345](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/345)
- Enable adjust max epoch of scheduler by `illian01` in [PR 350](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/350)
- Remove github action about hugging face space demo by `@illian01` in [PR 351](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/351)
- Update docs by `@illian01` in [PR 355](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/355), [PR 410](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/410)
- Backbone task compatibility checking refactoring by `@illian01` in [PR 361](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/361), [PR 364](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/364)
- Fix postprocessor return type as numpy.ndarray by `@illian01` in [PR 365](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/365)
- Update default asignees of issue template by `@illian01` in [PR 375](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/375)
- Refactoring: Remove CSV logger, change logger module input format by `@illian01` in [PR 377](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/377)
- Change ClassficationDataSampler logic by `@illian01` in [PR 382](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/382)
- Add YOLOX weights initialization step by `@illian01` in [PR 393](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/393)
- Minor update: detection postprocessor, dataset, and padding strategy by `@illian01` in [PR 395](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/395)
- Specify input size for onnx export and remove augmentation.img_size by `@illian01` in [PR 399](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/399)
- Update issue and pr template by `@illian01` in [PR 401](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/401)
- Add documentation auto deploy action by `@illian01` in [PR 405](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/405)

# v0.1.2

## New Features:

No changes to highlight.

## Bug Fixes:

- Remove union of int and list by `@illian01` in [PR 317](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/317)

## Breaking Changes:

No changes to highlight.

## Other Changes:

No changes to highlight.

# v0.1.1

## New Features:

- Enable customizing inference transform by `@illian01` in [PR 304](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/304)
- Add transform function: CenterCrop by `@illian01` in [PR 308](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/308)

## Bug Fixes:

- Fix automatic PIDNet weights download bug by `@illian01` in [PR 306](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/306)
- Resize default value to list by `@illian01` in [PR 315](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/315)

## Breaking Changes:

No changes to highlight.

## Other Changes:

- Update model caching directory and checkpoint configuration by `@deepkyu` in [PR 299](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/299), [PR 312](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/312)
- Minor docs update by `@illian01` in [PR 300](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/300)
- Update software development stage by `@illian01` in [PR 301](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/301)
- Fix size param of Resize to receive int or list by `@illian01` in [PR 310](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/310)
- Modify PIDNet conv bias, add head_list property on models by `@illian01` in [PR 311](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/311)

# v0.1.0

## New Features:

- Construct head by config file by `@illian01` in [PR 237](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/237)
- Construct neck by config file by `@illian01` in [PR 249](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/249)
- Add model: RetinaNet by `@illian01` in [PR 257](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/257)
- Select `gpus` with `environment` configuration by `@deepkyu` in [PR 269](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/269)
- Return logging directory path and fix training interfaces by `@deepkyu` in [PR 271](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/271)
- Add transform: AutoAugment by `@illian01` in [PR 281](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/281)

## Bug Fixes:

- Fix attribute error on fc by `@illian01` in [PR 252](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/252)
- Restore file export for stream log by `@deepkyu` in [PR 255](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/255)
- Fix CSV logging, configuration error, and misused loggings by `@deepkyu` in [PR 259](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/259)
- Fix minor bug in train.py by `@illian01` in [PR 277](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/277)
- Fix local classification dataset loader error by `@illian01` in [PR 279](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/279)
- Fix safetensors file overwriting bug by `@illian01` in [PR 289](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/289)
- Fix error on full model load by `@illian01` in [PR 295](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/295)

## Breaking Changes:

- Provide pytorch state dict with `.safetensors` and training summary with `.json` for a better utilization by `@deepkyu` in [PR 262](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/262)

## Other Changes:

- Refactoring for detection models by `@illian01` in [PR 260](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/260)
- Equalize logging format with `PyNetsPresso` by `@deepkyu` in [PR 263](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/263)
- Refactoring for clean docs by `@illian01` in [PR 265](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/265), [PR 266](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/266), [PR 272](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/272), [PR 273](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/273), [PR 274](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/274), [PR 284](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/284)
- Update docs up-to-date by `@illian01` in [PR 278](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/278)
- Refactoring model building code and move TASK_MODEL_DICT by `@illian01` in [PR 282](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/282)
- Add eps param on RMSprop by `@illian01` in [PR 285](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/285)
- Fix weights loading logic by `@illian01` in [PR 287](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/287), [PR 290](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/290)
- Change pretrained checkpoint name convention and update weight path and url by `@illian01` in [PR 291](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/291)
- Move seed field to environment config by `@illian01` in [PR 292](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/292)
- Move ResNet and Fc implementation code to core directory by `@illian01` in [PR 294](https://github.com/Nota-NetsPresso/netspresso-trainer/pull/294)

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
