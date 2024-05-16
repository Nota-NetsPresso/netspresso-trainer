python ./device_demo/demo.py\
    --model-path ./detection_yolox_s_graphmodule_epoch_450.tflite\
    --model-type original\
    --input-data 0\
    --score-thresh 0.4\
    --nms-thresh 0.65