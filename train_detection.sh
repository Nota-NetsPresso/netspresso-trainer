# python -m torch.distributed.launch --nproc_per_node 4 train.py --config config/efficientformer-detection.yaml
# python train.py --config config/efficientformer-detection.yaml
python train.py --data config/datasets/coco_yolo.yaml --config config/models/efficientformer-detection.yaml --training config/training/efficientformer.yaml