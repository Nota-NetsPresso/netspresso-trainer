<!-- FIXME: mostly copied from https://github.com/Nota-NetsPresso/PyNetsPresso/blob/main/README.md?plain=1 -->

### Convert Model and Benchmark the Converted Model
Convert an ONNX model into a TensorRT model, and benchmark the TensorRT model on the Jetson Nano.

```python
from loguru import logger
from netspresso.launcher import ModelConverter, ModelBenchmarker, ModelFramework, TaskStatus, DeviceName, SoftwareVersion

converter = ModelConverter(user_session=session)

model = converter.upload_model("./examples/sample_models/test.onnx")


conversion_task = converter.convert_model(
    model=model,
    input_shape=model.input_shape,
    target_framework=ModelFramework.TENSORRT,
    target_device_name=DeviceName.JETSON_AGX_ORIN,
    target_software_version=SoftwareVersion.JETPACK_5_0_1,
    wait_until_done=True
)

logger.info(conversion_task)

CONVERTED_MODEL_PATH = "converted_model.trt"
converter.download_converted_model(conversion_task, dst=CONVERTED_MODEL_PATH)


benchmarker = ModelBenchmarker(user_session=session)
benchmark_model = benchmarker.upload_model(CONVERTED_MODEL_PATH)
benchmark_task = benchmarker.benchmark_model(
    model=benchmark_model,
    target_device_name=DeviceName.JETSON_AGX_ORIN,
    target_software_version=SoftwareVersion.JETPACK_5_0_1,
    wait_until_done=True
)
logger.info(f"model inference latency: {benchmark_task.latency} ms")
logger.info(f"model gpu memory footprint: {benchmark_task.memory_footprint_gpu} MB")
logger.info(f"model cpu memory footprint: {benchmark_task.memory_footprint_cpu} MB")
```

### Available Devices with PyNetsPresso Launcher (Convert, Benchmark)

To fully use PyNetsPresso Launcher, model checkpoints from PyTorch has to be provided with [onnx] format. From our trainer, you can export the model checkpoint with onnx format when training is finished.  
With onnx files, the following devices are executable with PyNetsPresso Launcher:

- [Raspberry Pi]
  - 4 Mobel B
  - 3 Model B+
  - Zero W
  - Zero
- [Renesas Embedded AI MPUs]
  - RZ/V2L
  - RZ/V2M
- [NVIDIA Jetson]
  - Nano
  - TX2
  - Xavier
  - Nx
  - AGX Orin
- [AWS instance]
  - T4 

For more details, please refer to [compatibility matrix] in PyNetsPresso.

</br>
</br>


[onnx]: https://onnx.ai/
[Raspberry Pi]: https://www.raspberrypi.com/
[Renesas Embedded AI MPUs]: https://www.renesas.com/us/en/products/microcontrollers-microprocessors/rz-mpus/rzv-embedded-ai-mpus#explore
[NVIDIA Jetson]: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/
[AWS instance]: https://aws.amazon.com/pm/ec2/

[compatibility matrix]: https://github.com/Nota-NetsPresso/PyNetsPresso/tree/main#available-options-for-launcher-convert-benchmark