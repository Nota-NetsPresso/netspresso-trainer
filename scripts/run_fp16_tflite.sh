python ./device_demo/demo.py\ 
    --model-path ./detection_yolox_s_graphmodule_epoch_450.tflite\
    --dtype float\
    --input-data 0\
    --score-thresh 0.1\
    --nms-thresh 0.65