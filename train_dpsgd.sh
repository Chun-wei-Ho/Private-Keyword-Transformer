#!/bin/bash

# Train KWT on Speech commands v2 with 12 labels

set -euo pipefail

# source ./venv3/bin/activate

lang=en
clip_norm=10
noise_multiplier=1.46
delta=1e-5
. parse_options.sh

KWS_PATH=$PWD
# DATA_PATH=$KWS_PATH/data2
DATA_PATH=/home/chunwei/dataset/MLSW/$lang
MODELS_PATH=$KWS_PATH/models_data_v2_12_labels/kwt3
EXP=exp/${lang}_dpsgd_${noise_multiplier}_${clip_norm}
CMD_TRAIN="python -m kws_streaming.train.model_train_eval"

WANTED_WORD=`cut -d ' ' -f 1 $DATA_PATH/filtered/word_counts.txt | paste -sd,`
START_CHECKPOINT=$MODELS_PATH/best_weights

MODEL_ARGS="kws_transformer 
--num_layers 12
--heads 3
--d_model 192
--mlp_dim 768
--dropout1 0.
--attention_type time"

python MLSW/convert.py $START_CHECKPOINT $EXP/init.hdf5 $MODEL_ARGS

$CMD_TRAIN \
--dpsgd_norm_clip $clip_norm \
--dpsgd_delta $delta \
--dpsgd_noise_multiplier $noise_multiplier \
--wanted_words $WANTED_WORD \
--dataset_class 'MLSW_data.MLSWProcessor' \
--start_checkpoint $EXP/init.hdf5 \
--lang $lang \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $EXP/ \
--mel_upper_edge_hertz 7600 \
--optimizer 'dpsgd' \
--lr_schedule 'cosine' \
--how_many_training_steps '23438' \
--eval_step_interval 72 \
--warmup_epochs 10 \
--l2_weight_decay 0.1 \
--alsologtostderr \
--learning_rate '0.001' \
--batch_size 512 \
--label_smoothing 0.1 \
--window_size_ms 30.0 \
--window_stride_ms 10.0 \
--mel_num_bins 80 \
--dct_num_features 40 \
--resample 0.15 \
--train 1 \
--split 0 \
--use_spec_augment 1 \
--time_masks_number 2 \
--time_mask_max_size 25 \
--frequency_masks_number 2 \
--frequency_mask_max_size 7 \
--pick_deterministically 1 \
$MODEL_ARGS 2>&1 | tee $EXP/train.log

