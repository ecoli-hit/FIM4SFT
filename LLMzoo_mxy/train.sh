
export GPUS_PER_NODE=1
export MASTER_PORT=45556
#export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export SLURM_NNODES=1
export MASTER_PORT=9903
start_time="$(date "+%Y-%m-%d-%H-%M-%S")"

wandb offline

#export CUDA_VISIBLE_DEVICES=0,1

# /data/llx/llama-7b-hf/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348

# /data/llx/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9

# --num_gpus $GPUS_PER_NODE --num_nodes $SLURM_NNODES

CUDA_VISIBLE_DEVICES=0 python train_random.py \
    --model_name_or_path "/data/lyj/hf_models/llama-2-7b-hf" \
    --model_max_length 2048 \
    --data_path /data/mxy/Finstruction/data/code/commitpackft/data/c/data.jsonl \
    --output_dir /data/mxy/Finstruction/output/c/python \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 500 \
    --evaluation_strategy "no" \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \

# nohup bash train.sh  >> llama2_7b_LIMA.log 2>&1 &

# (sleep 2h ; nohup bash train.sh  >> llama2_7b_lr2e-5_bs16_baseline_apaca_gpt4_total_32681_5times_std.log 2>&1 &) &

# (sleep 5s ; date) &

