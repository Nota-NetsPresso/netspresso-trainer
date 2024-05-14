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
    parser.add_argument('--input-data', type=str, required=True, dest='input_data', help="Put 0 if you want tou use cam. Or put image data directory path.")
    parser.add_argument('--score-thresh', type=float, required=True, dest='score_thresh', help="Score threshold to discard boxes with low score")
    parser.add_argument('--nms-thresh', type=float, required=True, dest='nms_thresh', help="IoU threshold for non-maximum suppression")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cpu') # Only support cpu mode.
    model_type = 'tflite'

    # Load dataset
    cam_mode = args.input_data == '0'
    if cam_mode:
        dataset = LoadCamera()
    else:
        dataset = LoadDirectory('./data/WiderPerson-yolo/images/test')
        save_dir = Path('./outputs/device_demo')
        os.makedirs(save_dir, exist_ok=True)

    # Load model
    if model_type == 'tflite':
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                raise ImportError("Failed to import tensorflow lite. Please install tflite_runtime or tensorflow")

        model = tflite.Interpreter(model_path=args.model_path, num_threads=4)
        model.allocate_tensors()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load postprocessor
    postprocessor = DetectionPostprocessor(score_thresh=args.score_thresh, nms_thresh=args.nms_thresh, class_agnostic=False)
    quantization = True if model.get_input_details()[0]['dtype'] == np.int8 else False

    # Warmup model
    dummy_image = torch.empty((1, 320, 320, 3), device=device)
    if quantization: 
        dummy_image = dummy_image.to(torch.int8)
        input_scale, input_zero_point = model.get_input_details()[0]['quantization']
    model.set_tensor(0, dummy_image)
    for _ in range(5):
        model.invoke()

    # Inference
    for i, original_img in enumerate(dataset):
        timer = TimeRecode()
        # Preprocess
        preprocess_time = TimeRecode()
        tensor_img = preprocess(original_img, size=320)

        if quantization:
            tensor_img = tensor_img / input_scale + input_zero_point
            tensor_img = tensor_img.to(torch.int8)

        preprocess_time.end()
        # Model forward
        forward_time = TimeRecode()
        model.set_tensor(0, tensor_img)
        model.invoke()

        output_info = [(details['index'], details['quantization_parameters']) for details in model.get_output_details()]
        output_info.sort()

        output = [torch.tensor(model.get_tensor(index)).permute(0, 3, 1, 2) for index, _ in output_info]

        if quantization:
            for i, (_, quantization_parameters) in enumerate(output_info):
                input_scale = quantization_parameters['scales']
                input_zero_point = quantization_parameters['zero_points']
                output[i] = (output[i] - input_zero_point.astype('int8')).to(torch.float32) * input_scale

            output_1 = torch.cat([output[1], output[2], output[0]], dim=1)
            output_2 = torch.cat([output[4], output[5], output[3]], dim=1)
            output_3 = torch.cat([output[7], output[8], output[6]], dim=1)
            output = [output_3, output_2, output_1]

        output = {'pred': output}

        forward_time.end()
        # Postprocess
        postprocess_time = TimeRecode()
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

        postprocess_time.end()
        timer.end()
        fps = f'{int(np.round(1 / timer.elapsed))} FPS' # frame per second
        text_size, _ = cv2.getTextSize(str(fps), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        save_img = cv2.putText(save_img, str(fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        print(f'Preprocess time: {preprocess_time.elapsed}. Forward time: {forward_time.elapsed}. Postprocess time: {postprocess_time.elapsed}')

        if cam_mode:
            cv2.imshow('window', save_img[..., ::-1])
            cv2.waitKey(1)
        else:
            save_img = Image.fromarray(save_img)
            save_img.save(save_dir / f'{i}.png')
