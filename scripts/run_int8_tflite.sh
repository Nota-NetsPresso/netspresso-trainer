python ./device_demo/demo.py\ 
    --model-path ./detection_yolox_s_graphmodule_epoch_450_int8.tflite\
    --dtype int8\
    --input-data 0\
    --score-thresh 0.1\
    --nms-thresh 0.65