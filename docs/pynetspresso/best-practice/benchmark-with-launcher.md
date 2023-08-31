
## ResNet classification model with PyNetsPresso Launcher

### Train your ResNet50 model

Train your model with your own dataset. 

### With PyNetsPresso Launcher

Please write the path of your onnx checkpoint from training result, and replace the example checkpoint path with your file:

``` python
model = converter.upload_model("YOUR/CHECKPOINT/PATH")
```

Please refer to [PyNetsPresso Launcher Guide](../launcher.md) for more details.
  
### Benchmark Result

The following is an example result tested with Jetson AGX Orin with TensorRT backend:

```
root@ee7398d3c015:/home/appuser/netspresso-trainer/np-compatibility# python tools/pynetspresso.py 
2023-08-28 03:34:44.206 | INFO     | netspresso.client.config:<module>:10 - Read prod config
2023-08-28 03:34:44.566 | INFO     | netspresso.client:__login:50 - Login successfully
2023-08-28 03:34:45.082 | INFO     | netspresso.client:__get_user_info:67 - successfully got user information
2023-08-28 03:35:05.647 | INFO     | netspresso.launcher:convert_model:104 - Converting Model for Jetson-AGX-Orin (tensorrt)
2023-08-28 03:36:24.818 | INFO     | __main__:<module>:20 - user_uuid='d3aa4f44-2237-4a4a-8c74-ee58e43d9472' input_model_uuid='d8727c33-2b56-4b76-ac79-2f71c579951c' status='FINISHED' input_shape=InputShape(batch=1, channel=3, input_size='256, 256') data_type='FP16' software_version='5.0.1-b118' framework='onnx' convert_task_uuid='ef8eea24-9def-4bfe-80d0-9b48986f62f6' output_model_uuid='15d4d3d1-6ac1-4e3e-acc9-cfe997184337' model_file_name='model.trt' target_device_name='Jetson-AGX-Orin'
2023-08-28 03:36:36.240 | INFO     | netspresso.launcher:download_converted_model:171 - The file has been successfully downloaded at : converted_model.trt
2023-08-28 03:39:54.680 | INFO     | __main__:<module>:34 - model inference latency: 1.2281 ms
2023-08-28 03:39:54.682 | INFO     | __main__:<module>:35 - model gpu memory footprint: 49.0 MB
2023-08-28 03:39:54.683 | INFO     | __main__:<module>:36 - model cpu memory footprint: 302.0 MB
```

<br/>
<br/>