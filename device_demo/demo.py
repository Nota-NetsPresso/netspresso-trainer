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
BOX_COLOR = (0, 255, 255)


def parse_args():
    parser = argparse.ArgumentParser(description="Parser for NetsPresso demo")

    parser.add_argument('--model-path', type=str, required=True, dest='model_path', help="Path to the model")
    parser.add_argument('--input-data', type=str, required=True, dest='input_data', help="Put 0 if you want tou use cam. Or put image data directory path.")
    parser.add_argument('--score-thresh', type=float, required=True, dest='score_thresh', help="Score threshold to discard boxes with low score")
    parser.add_argument('--nms-thresh', type=float, required=True, dest='nms_thresh', help="IoU threshold for non-maximum suppression")
    parser.add_argument('--img-size', type=int, default=320, dest='img_size', help="Input image size")

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
    postprocessor = DetectionPostprocessor(score_thresh=args.score_thresh, nms_thresh=args.nms_thresh)
    quantization = True if model.get_input_details()[0]['dtype'] == np.int8 else False

    # Warmup model
    dummy_image = torch.empty((1, args.img_size, args.img_size, 3), device=device)
    if quantization: 
        dummy_image = dummy_image.to(torch.int8)
    model.set_tensor(0, dummy_image)
    for _ in range(5):
        model.invoke()

    # Inference
    for i, original_img in enumerate(dataset):
        timer = TimeRecode()
        # Preprocess
        input_img = preprocess(original_img, size=args.img_size)

        if quantization:
            input_scale, input_zero_point = model.get_input_details()[0]['quantization']
            input_img = input_img / input_scale + input_zero_point
            input_img = input_img.astype('int8')

        # Model forward
        model.set_tensor(0, input_img)
        model.invoke()

        output_info = [(details['index'], details['quantization_parameters']) for details in model.get_output_details()]
        output_info.sort()

        output = [np.transpose(model.get_tensor(index), (0, 3, 1, 2)) for index, _ in output_info]

        if quantization:
            for i, (_, quantization_parameters) in enumerate(output_info):
                input_scale = quantization_parameters['scales']
                input_zero_point = quantization_parameters['zero_points']
                output[i] = (output[i].astype('float32') - input_zero_point.astype('float32')) * input_scale.astype('float32')

            output_1 = np.concatenate([output[1], output[2], output[0]], axis=1)
            output_2 = np.concatenate([output[4], output[5], output[3]], axis=1)
            output_3 = np.concatenate([output[7], output[8], output[6]], axis=1)
            output = [output_3, output_2, output_1]

        output = {'pred': output}

        # Postprocess
        detections = postprocessor(output, original_shape=(args.img_size, args.img_size))
        detections = detections[0]

        keeps = np.logical_or(detections[1] == 0, detections[1] == 2) # Only draw "pedestrians" and "partially-visible persons"
        detections = (detections[0][keeps], detections[1][keeps])

        resize_factor = max((original_img.shape[0] / args.img_size), (original_img.shape[1] / args.img_size))
        detections[0][:, 1::2] *= resize_factor
        detections[0][:, :4:2] *= resize_factor

        # Visualize
        save_img = original_img
        for bbox_label, class_label in zip(detections[0], detections[1]):
            # unnormalize depending on the visualizing image size
            x1 = int(bbox_label[0])
            y1 = int(bbox_label[1])
            x2 = int(bbox_label[2])
            y2 = int(bbox_label[3])

            save_img = cv2.rectangle(save_img, (x1, y1), (x2, y2), color=BOX_COLOR, thickness=2)

        timer.end()
        fps = f'{int(np.round(1 / timer.elapsed))} FPS' # frame per second
        text_size, _ = cv2.getTextSize(str(fps), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        save_img = cv2.putText(save_img, str(fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if cam_mode:
            cv2.imshow('window', save_img[..., ::-1])
            cv2.waitKey(1)
        else:
            save_img = Image.fromarray(save_img)
            save_img.save(save_dir / f'{i}.png')
