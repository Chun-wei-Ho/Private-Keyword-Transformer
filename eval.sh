#!/bin/bash

set -euo pipefail

LANG=en
. parse_options.sh

DATA_PATH=/home/chunwei/dataset/MLSW/$LANG
MODELS_PATH=exp/$LANG
CMD_TRAIN="python -m kws_streaming.train.model_train_eval"
WANTED_WORD=`cut -d ' ' -f 1 $DATA_PATH/filtered/word_counts.txt | paste -sd,`

mkdir -p exp

$CMD_TRAIN \
--wanted_words $WANTED_WORD \
--data_url '' \
--data_dir $DATA_PATH/ \
--dataset_class 'MLSW_data.MLSWProcessor' \
--train_dir $MODELS_PATH \
--mel_upper_edge_hertz 7600 \
--optimizer 'adamw' \
--lang $LANG \
--lr_schedule 'cosine' \
--how_many_training_steps '1' \
--eval_step_interval 72 \
--warmup_epochs 10 \
--l2_weight_decay 0.1 \
--learning_rate '0.001' \
--batch_size 512 \
--label_smoothing 0.1 \
--window_size_ms 30.0 \
--window_stride_ms 10.0 \
--mel_num_bins 80 \
--dct_num_features 40 \
--resample 0.15 \
--alsologtostderr \
--train 0 \
--split 0 \
--use_spec_augment 1 \
--time_masks_number 2 \
--time_mask_max_size 25 \
--frequency_masks_number 2 \
--frequency_mask_max_size 7 \
--pick_deterministically 1 \
kws_transformer \
--num_layers 12 \
--heads 3 \
--d_model 192 \
--mlp_dim 768 \
--dropout1 0. \
--attention_type 'time' \
