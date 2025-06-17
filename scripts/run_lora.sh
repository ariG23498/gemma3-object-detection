
#!bin/bash


python finetune.py \
  --model_id google/gemma-3-4b-pt \
  --dataset_id ariG23498/license-detection-paligemma \
  --batch_size 8 \
  --lr 2e-5 \
  --checkpoint_id oreonmayo/gemma-3-4b-pt-object-detection-aug \
  --peft_type lora \
  --peft_config peft_configs/lora_configs.yaml 