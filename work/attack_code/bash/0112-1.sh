#!/bin/bash
export FLAGS_fraction_of_gpu_memory_to_use=0.5

python main.py\
     --verbose 1\
     --models ResNeXt50_32x4d MobileNetV2_x2_0 ResNeXt_Ens32 InceptionV4 ResNet50 EfficientNetB0\
     --weights 0.1 0.05 0.4 0.35 0.05 0.05\
     --step_size 0.070\
     --norm_regulizer 0.0\
     --clip_eps 0.99\
     --attack_iters 20\
     --attack_confidence 1.1\
     --output ../0112-1_result/\
     --sparse_percentage 95\
     --diversity_iter 10\
     --momentum_decay 0.9\
     --lr_decay 0.78\
     --start 0\
     --end 60 &\

python main.py\
     --verbose 1\
     --models ResNeXt50_32x4d MobileNetV2_x2_0 ResNeXt_Ens32 InceptionV4 ResNet50 EfficientNetB0\
     --weights 0.1 0.05 0.4 0.35 0.05 0.05\
     --step_size 0.070\
     --norm_regulizer 0.0\
     --clip_eps 0.99\
     --attack_iters 20\
     --attack_confidence 1.1\
     --output ../0112-1_result/\
     --sparse_percentage 95\
     --diversity_iter 10\
     --momentum_decay 0.9\
     --lr_decay 0.78\
     --start 60\
     --end 120\