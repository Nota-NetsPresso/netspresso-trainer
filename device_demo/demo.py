import argparse
from pathlib import Path
import os

import numpy as np
import cv2
import torch
from PIL import Image

from dataloader import LoadDirectory, LoadCamera, preprocess
from postprocessor import DetectionPostprocessor
from utils import TimeRecode


CLASS_MAP = ["pedestrians", "riders", "partially-visible persons", "ignore regions", "crowd"]

def _voc_color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def parse_args():
    parser = argparse.ArgumentParser(description="Parser for NetsPresso demo")

    parser.add_argument('--model-path', type=str, required=True, dest='model_path', help="Path to the model")
    parser.add_argument('--score-thresh', type=float, required=True, dest='score_thresh', help="Score threshold to discard boxes with low score")
    parser.add_argument('--nms-thresh', type=float, required=True, dest='nms_thresh', help="IoU threshold for non-maximum suppression")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args.model_path)

    task = 'detection'
    distributed = False

    device = torch.device('cpu') # Only support cpu mode.
    model_type = 'tflite'

    # Load dataset
    #dataset = LoadDirectory('./data/WiderPerson-yolo/images/test')
    dataset = LoadCamera()

    # Load model
    if model_type == 'tflite':
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                raise ImportError("Failed to import tensorflow lite. Please install tflite_runtime or tensorflow")

        model = tflite.Interpreter(model_path=args.model_path)
        model.allocate_tensors()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load postprocessor
    postprocessor = DetectionPostprocessor(score_thresh=args.score_thresh, nms_thresh=args.nms_thresh, class_agnostic=False)

    # Warmup model
    dummy_image = torch.empty((1, 320, 320, 3), dtype=torch.float, device=device)
    model.set_tensor(0, dummy_image)
    model.invoke()

    # Inference
    save_dir = Path('./outputs/device_demo')
    os.makedirs(save_dir, exist_ok=True)
   
    for i, original_img in enumerate(dataset):
        timer = TimeRecode()
        # Preprocess
        tensor_img = preprocess(original_img, size=320)

        # Model forward
        model.set_tensor(0, tensor_img)
        model.invoke()

        output_index = [details['index'] for details in model.get_output_details()]
        output_index.sort()
        output_index = output_index[::-1]
        output = {'pred': [torch.tensor(model.get_tensor(index)).permute(0, 3, 1, 2) for index in output_index]}

        # Postprocess
        detections = postprocessor(output, original_shape=(320, 320))
        detections = detections[0]

        resize_factor = max((original_img.size[1] / 320), (original_img.size[0] / 320))
        detections[0][:, 1::2] *= resize_factor
        detections[0][:, :4:2] *= resize_factor

        # Visualize
        save_img = np.array(original_img)
        cmap = _voc_color_map(len(CLASS_MAP))
        for bbox_label, class_label in zip(detections[0], detections[1]):
            class_name = CLASS_MAP[class_label]

            # unnormalize depending on the visualizing image size
            x1 = int(bbox_label[0])
            y1 = int(bbox_label[1])
            x2 = int(bbox_label[2])
            y2 = int(bbox_label[3])
            color = cmap[class_label].tolist()

            save_img = cv2.rectangle(save_img, (x1, y1), (x2, y2), color=color, thickness=2)
            text_size, _ = cv2.getTextSize(str(class_name), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_w, text_h = text_size
            save_img = cv2.rectangle(save_img, (x1, y1-5-text_h), (x1+text_w, y1), color=color, thickness=-1)
            save_img = cv2.putText(save_img, str(class_name), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        timer.end()
        fps = f'{int(np.round(1 / timer.elapsed))} FPS' # frame per second
        text_size, _ = cv2.getTextSize(str(fps), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        save_img = cv2.putText(save_img, str(fps), (0, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        save_img = Image.fromarray(save_img)
        save_img.save(save_dir / f'{i}.png')

    print()