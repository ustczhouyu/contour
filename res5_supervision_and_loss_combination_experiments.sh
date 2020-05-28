#!/bin/bash

# Binray with and without res5 supervision, uncertainty
python train_net.py --config-file configs/cityscapes_panoptic_contour_binary_seperate_R_50_1x_uncertainty.yaml --num-gpus 2
wait
python train_net.py --config-file configs/cityscapes_panoptic_contour_binary_seperate_R_50_1x_res5supervision.yaml --num-gpus 2
wait
python train_net.py --config-file configs/cityscapes_panoptic_contour_binary_seperate_R_50_1x_uncertainty_res5supervision.yaml --num-gpus 2
wait

# Multiclass with and without res5 supervision, uncertainty

python train_net.py --config-file configs/cityscapes_panoptic_contour_multi_seperate_R_50_1x.yaml --num-gpus 2
wait
python train_net.py --config-file configs/cityscapes_panoptic_contour_multi_seperate_R_50_1x_uncertainty.yaml --num-gpus 2
wait
python train_net.py --config-file configs/cityscapes_panoptic_contour_multi_seperate_R_50_1x_res5supervision.yaml --num-gpus 2
wait
python train_net.py --config-file configs/cityscapes_panoptic_contour_multi_seperate_R_50_1x_uncertainty_res5supervision.yaml --num-gpus 2
