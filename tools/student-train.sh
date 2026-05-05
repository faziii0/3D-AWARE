#!/bin/bash

#Train RCNN with teacher-student setup
CUDA_VISIBLE_DEVICES=0,1 python train_rcnn.py \
--cfg_file cfgs/student.yaml \
--teacher_cfg_file cfgs/teacher.yaml \
--batch_size 1 \
--train_mode rcnn_online \
--epochs 50 \
--ckpt_save_interval 1 \
--output_dir log/student/ \
--data_path /media/home/SSD/3D-AWARE/data \
--teacher_ckpt /media/home/SSD/3D-AWARE/datatools/log/teacher/ckpt/checkpoint_epoch_43.pth \
--mgpus \
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




