python ./device_demo/demo.py\
    --model-path detection_yolox_s_graphmodule_epoch_450_head_modified_int8.tflite\
    --input-data 0\
    --score-thresh 0.4\
    --nms-thresh 0.65