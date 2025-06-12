!#bin/bash


python train.py \
  --model_id google/gemma-3-4b-pt \
  --dataset_id ariG23498/license-detection-paligemma \
  --batch_size 8 \
  --lr 2e-5 \
  --checkpoint_id oreonmayo/gemma-3-4b-pt-object-detection-aug