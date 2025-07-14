#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
nnUNetv2_train 3 2d all -tr nnUNetTrainerUMambaEnc