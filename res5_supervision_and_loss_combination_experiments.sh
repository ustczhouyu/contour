#!/bin/bash

python train_net.py --config-file configs/cityscapes_panoptic_contour_binary_seperate_R_50_1x_bce.yaml --num-gpus 2 --resume #--eval-only
wait
python train_net.py --config-file configs/cityscapes_panoptic_contour_binary_seperate_R_50_1x_bce_dice.yaml --num-gpus 2 --resume #--eval-only
wait
python train_net.py --config-file configs/cityscapes_panoptic_contour_binary_seperate_R_50_1x_bce_dice_huber.yaml --num-gpus 2 --resume #--eval-only
