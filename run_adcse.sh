#!/bin/bash

python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/AdCSE \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --moco_m 0.995 \
    --moco_t 0.05 \
    --mem_m 0.9 \
    --mem_t 0.05 \
    --mem_lr 3e-3 \
    --mem_wd 1e-4 \
    --neg_num 64 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"