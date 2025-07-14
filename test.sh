#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
nnUNetv2_predict -i ../data/nnUNet_raw/Dataset004_Fives/imagesTs/ -o ../data/nnUNet_results/Dataset004_Fives/nnUNetTrainerUMambaEnc__nnUNetPlans__2d/predict_xyzw/349checkpoint_latest -d 4 -c 2d -f all -tr nnUNetTrainerUMambaEnc -chk 349checkpoint_latest.pth --disable_tta