# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

import argparse
import numpy as np
from omegaconf import OmegaConf

from dataloaders.cam_loader import LoadCamera


def parse_args():
    parser = argparse.ArgumentParser(description="Parser for device demo")
    parser.add_argument('--config-path', type=str, required=True, dest='config_path', help="Path to the demo configuration file")
    return parser.parse_args()


def import_modules_by_task(conf):
    if conf.task == 'detection':
        from visualizers.detection_visualizer import DetectionVisualizer as Visualizer
        from preprocessor import Preprocessor
        from postprocessors.detection_postprocessor import DetectionPostprocessor as Postprocessor
    else:
        raise NotImplementedError(f"Task {args.task} is not supported yet")

    return Preprocessor, Postprocessor, Visualizer


def load_model(model_path):
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        try:
            import tensorflow.lite as tflite
        except ImportError:
            raise ImportError("Failed to import tensorflow lite. Please install tflite_runtime or tensorflow")
    interpreter = tflite.Interpreter(model_path=model_path, num_threads=4) # TODO: Get num_threads from config or environment
    interpreter.allocate_tensors()
    return interpreter


def forward_model(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Only one input tensor (image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # TODO: This is set to only detection task. Need to generalize
    output_info = [details['index'] for details in output_details]
    output = [np.transpose(model.get_tensor(index), (0, 3, 1, 2)) for index in output_info] # (b, h, w, c) -> (b, c, h, w)
    return output


if __name__ == '__main__':
    args = parse_args()

    conf = OmegaConf.load(args.config_path).runtime
    Preprocessor, Postprocessor, Visualizer = import_modules_by_task(conf)

    model = load_model(conf.model_path)
    preprocessor = Preprocessor(conf.preprocess)
    postprocessor = Postprocessor(conf.postprocess)
    visualizer = Visualizer()

    dataloader = LoadCamera()
    for img in dataloader:
        img_draw = img.squeeze()

        img = preprocessor(img)
        model_input_shape = img.shape[1:3]
        output = forward_model(model, img)
        pred = postprocessor({'pred': output}, model_input_shape)[0]

        img_draw = visualizer.draw(img_draw, pred, model_input_shape)
        img_draw = img_draw[..., ::-1] # RGB -> BGR to use cv2 imshow
        visualizer.visualize(img_draw)
