#!/usr/bin/env bash
set -e

gpu_id=0
ckt_path="weights/pretrained_visa_colondb_86.pth"
sd_config_path="method/v1-inference.yaml"
sd_ckpt_path=""
sd_vae_ckpt_path=""

datasets=(
  uni_medical
  brain_mri
  btad
  clinicdb
  dagm
  headct
  isic
  kvasir
  tn3k
  endo
  br35h
  brain
  camelyon16
  braTS2021
  covid
)

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

for dataset_name in "${datasets[@]}"; do
  CUDA_VISIBLE_DEVICES=$gpu_id python test.py \
    --testing_model dataset \
    --ckt_path "$ckt_path" \
    --save_fig True \
    --testing_data "$dataset_name" \
    "${common_sd_args[@]}"
done
