#!/bin/bash
export NCCL_P2P_DISABLE="1"
ENTITY="qkrwlgh0314"
PROJECT="levae"
# WANDB_KEY="4ab8d4a0db9aec6c8095"
WANDB_KEY="4ab8d4a0db9aec6c80956ccf58616de15392a463"

# export TORCH_DISTRIBUTED_DEBUG=DETAIL

EXP_NAME=0512_debug
EXPS_DIR=exps
# CONFIG=configs/default.yaml
CONFIG=$EXPS_DIR/$EXP_NAME/config.yaml
DATA_PATH=/dataset/imagenet/train
NPROC=2

torchrun --standalone --nproc_per_node=$NPROC train_vae.py \
    --exps_dir $EXPS_DIR \
    --exp_name $EXP_NAME \
    --config $CONFIG \
    --data_path $DATA_PATH \
    --last_ckpt_every 1000 \
    --ckpt_every 10000 \
    --log_every 10 \
    --visualize_every 1000 \
    --save_visualizations_locally \
    --num_workers 8 \
    --seed 42 \
    # --wandb \
    # --wandb_key $WANDB_KEY \
    # --wandb_entity $ENTITY \
    # --wandb_project $PROJECT
    # --resume_last

# sbatch -p base_suma_rtx3090 -q base_qos --gres=gpu:4 train.sh
# sbatch -p big_suma_rtx3090 -q big_qos --gres=gpu:4 train.sh
# sbatch -p gigabyte_a5000 -q big_qos --gres=gpu:4 train.sh
# sbatch -p suma_rtx4090 -q big_qos --gres=gpu:4 train.sh
# sbatch -p gigabyte_A6000 -q base_qos --gres=gpu:2 train.sh
# sbatch -p base_suma_rtx3090 -q base_qos --gres=gpu:4 train.sh
# sbatch -p dell_rtx3090 -q big_qos --gres=gpu:4 train.sh
# sbatch -p big_suma_rtx3090 -q big_qos --gres=gpu:4 train.sh