#!/bin/bash

# Train KWT on Speech commands v2 with 12 labels

set -euo pipefail

# source ./venv3/bin/activate

lang=fr
adapter_dim=192

. parse_options.sh

SOURCE_PATH=/home/chunwei/dataset/google-speech-command_v2
KWS_PATH=$PWD
SOURCE_MODEL_PATH=$KWS_PATH/models_data_v2_12_labels/kwt3
TARGET_MODEL_PATH=exp/${lang}_pate_students_50_7.96_adapter_${adapter_dim}
EXP=exp/evaluate_`basename $TARGET_MODEL_PATH`
CMD_TRAIN="python -m kws_streaming.train.model_train_eval"

START_CHECKPOINT1=$SOURCE_MODEL_PATH/best_weights
START_CHECKPOINT2=$TARGET_MODEL_PATH/best_weights

MODEL_ARGS="kws_transformer 
--num_layers 12
--heads 3
--d_model 192
--mlp_dim 768
--dropout1 0.
--attention_type time
--adapter_dim $adapter_dim
"

EVAL_ARGS="--start_checkpoint $EXP/init.hdf5
    --lang $lang
    --dataset_class input_data.AudioProcessor
    --train_dir $EXP/
    --mel_upper_edge_hertz 7600
    --optimizer adamw
    --lr_schedule cosine
    --data_dir $SOURCE_PATH/
    --eval_step_interval 20
    --alsologtostderr
    --batch_size 512
    --label_smoothing 0.1
    --window_size_ms 30.0
    --window_stride_ms 10.0
    --mel_num_bins 80
    --dct_num_features 40
    --resample 0.15
    --train 0
    --split 1
    --use_spec_augment 0
    --pick_deterministically 1"

mkdir -p $EXP
python MLSW/convert.py  \
            --save_weights_only \
            --input_checkpoint2 $START_CHECKPOINT2 \
            $START_CHECKPOINT1 $EXP/best_weights $MODEL_ARGS

$CMD_TRAIN $EVAL_ARGS $MODEL_ARGS 2>&1 | tee $EXP/eval.log

