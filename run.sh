#!/bin/bash
python train_net.py --config-file configs/cityscapes_panoptic_contour_binary_seperate_R_50_1x_res5supervision.yaml --num-gpus 2
wait
python train_net.py --config-file configs/cityscapes_panoptic_contour_binary_seperate_R_50_1x.yaml --num-gpus 2
