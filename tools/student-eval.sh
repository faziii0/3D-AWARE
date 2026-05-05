#!/bin/bash

################################## eval CAR

CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py \
--eval_mode rcnn_online  \
--eval_all \
--cfg_file cfgs/student.yaml \
--batch_size 2 \
--output_dir ./log/student/eval_results \
--ckpt_dir ./log/student/ckpt \
--data_path /media/home/SSD/3D-AWARE/data \
--set \
LI_FUSION.ENABLED True \
LI_FUSION.ADD_Image_Attention True \
USE_IM_DEPTH False \
CROSS_FUSION True \
USE_P2I_GATE True \
USE_IMAGE_LOSS True \
IMAGE_WEIGHT 1.0 \
USE_IMAGE_SCORE True \
USE_IMG_DENSE_LOSS True \
USE_MC_LOSS True \
MC_LOSS_WEIGHT 1.0 \
I2P_Weight 0.5 \
P2I_Weight 0.5 \
ADD_MC_MASK True \
MC_MASK_THRES 0.2 \
SAVE_MODEL_PREP 0.8


# ################################## eval CAR--test

# CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py \
# --eval_mode rcnn_online  \
# --cfg_file cfgs/student.yaml \
# --batch_size 2 \
# --output_dir ./log/student/eval_results-test-47 \
# --ckpt ./log/student/ckpt/checkpoint_epoch_47.pth \
# --data_path /media/home/SSD/3D-AWARE/data \
# --test \
# --set \
# LI_FUSION.ENABLED True \
# LI_FUSION.ADD_Image_Attention True \
# USE_IM_DEPTH False \
# CROSS_FUSION True \
# USE_P2I_GATE True \
# USE_IMAGE_LOSS True \
# IMAGE_WEIGHT 1.0 \
# USE_IMAGE_SCORE True \
# USE_IMG_DENSE_LOSS True \
# USE_MC_LOSS True \
# MC_LOSS_WEIGHT 1.0 \
# I2P_Weight 0.5 \
# P2I_Weight 0.5 \
# ADD_MC_MASK True \
# MC_MASK_THRES 0.2 \
# SAVE_MODEL_PREP 0.8
