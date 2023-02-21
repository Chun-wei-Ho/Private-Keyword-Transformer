#!/bin/bash

# Train KWT on Speech commands v2 with 12 labels

set -euo pipefail

# source ./venv3/bin/activate

lang=en
nb_teachers=50
nepochs=80
batch_size=512
. parse_options.sh

KWS_PATH=$PWD
# DATA_PATH=$KWS_PATH/data2
DATA_PATH=/home/chunwei/dataset/MLSW/$lang
EXP=exp/${lang}_pate_teachers_${nb_teachers}_from_scratch
CMD_TRAIN="python -m kws_streaming.train.model_train_eval"

NUM_TRAIN=`wc -l $DATA_PATH/filtered/${lang}_train.csv | cut -d ' ' -f 1`
NUM_TRAIN=`echo ${NUM_TRAIN} - 1 | bc`
NUM_STEPS=`echo ${NUM_TRAIN} \* $nepochs / ${batch_size} / ${nb_teachers} | bc`

WANTED_WORD=`cut -d ' ' -f 1 $DATA_PATH/filtered/word_counts.txt | paste -sd,`

MODEL_ARGS="kws_transformer 
--num_layers 12
--heads 3
--d_model 192
--mlp_dim 768
--dropout1 0.
--attention_type time
"
# --reprogram trainable_noise


for teacher_id in $(seq 0 $(($nb_teachers - 1))); do
    if [ ! -f $EXP/$teacher_id/.done ]; then
        mkdir $EXP/$teacher_id
        $CMD_TRAIN \
        --wanted_words $WANTED_WORD \
        --dataset_class 'MLSW_data.MLSWProcessor' \
        --lang $lang \
        --data_url '' \
        --data_dir $DATA_PATH/ \
        --train_dir $EXP/$teacher_id/ \
        --mel_upper_edge_hertz 7600 \
        --optimizer 'adamw' \
        --lr_schedule 'cosine' \
        --nb_teachers $nb_teachers \
        --teacher_id $teacher_id \
        --how_many_training_steps $NUM_STEPS \
        --eval_step_interval 20 \
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
        $MODEL_ARGS 2>&1 | tee  $EXP/$teacher_id/train.log
        touch $EXP/$teacher_id/.done
    fi
done

