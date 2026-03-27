#!/usr/bin/env bash
set -e

gpu_id=0
sd_config_path="method/v1-inference.yaml"
sd_ckpt_path=""
sd_vae_ckpt_path=""

common_sd_args=(
  --use_sd_feature True
  --sd_config_path "$sd_config_path"
  --sd_timestep 50
  --sd_base_size 512
  --sd_use_attention True
)

if [ -n "$sd_ckpt_path" ]; then
  common_sd_args+=(--sd_ckpt_path "$sd_ckpt_path")
fi

if [ -n "$sd_vae_ckpt_path" ]; then
  common_sd_args+=(--sd_vae_ckpt_path "$sd_vae_ckpt_path")
fi

# Note: Since we have utilized half-precision (FP16) for training, the training process can occasionally be unstable.
# It is recommended to run the training process multiple times and choose the best model based on performance
# on the validation set as the final model.

# pre-trained on MVtec and colondb
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --save_fig True --training_data mvtec colondb --testing_data visa "${common_sd_args[@]}"

# pre-trained on Visa and Clinicdb
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --save_fig True --training_data visa clinicdb --testing_data mvtec "${common_sd_args[@]}"

# This model is pre-trained on all available data to create a powerful Zero-Shot Anomaly Detection (ZSAD) model for demonstration purposes.
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --save_fig True \
  --training_data \
  br35h brain_mri btad clinicdb colondb \
  dagm dtd headct isic mpdd mvtec sdd tn3k visa \
  --testing_data mvtec \
  "${common_sd_args[@]}"
