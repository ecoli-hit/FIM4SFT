
# export GPUS_PER_NODE=1
export MASTER_PORT=45556
#export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# export SLURM_NNODES=1
# export MASTER_PORT=9904
# start_time="$(date "+%Y-%m-%d-%H-%M-%S")"
ROOT=/data/mxy/Finstruction
MODEL="/data/mxy/models/llama2-7b-hf"


CUDE_VISIBLE_DEVICES=2,3

# don't assign path and dir both, exact path for 1 dataset and for cancate use dir,
# when use "super_ni" dataset, should assign dir for all tasks and merge file for task types and 
# the tasks to load

# TRAIN_PATH=$ROOT/data/train/super_ni_convert/fs_3/task022_cosmosqa_passage_inappropriate_binary
# EVAL_PATH=$ROOT/data/train/super_ni_convert/fs_3/task022_cosmosqa_passage_inappropriate_binary

OUTPUT_DIR=$ROOT/FIM_pool/test5

# "super_ni" "hybrid" "single"
DATA_TYPE=single
# The config used for all data type
DATA_PATH_DIR=/data/mxy/Finstruction/data/train/super_ni_convert/fs_3/task022_cosmosqa_passage_inappropriate_binary
# The tasks you want to use in super_ni training
MERGE_FILE=/data/mxy/Finstruction/data/train/super_ni_convert/merge.json
TASK_LIST='Preposition_Prediction'
# The config use to hybrid, the total dataset to load
DATA_CONFIG=/data/mxy/Finstruction/FIM_pool/test/data_config.json

# When epochs==1, automatically dump fisher information matrix and won't save model 
# please set "SAVE" as "no"
# for training set "SAVE" as "steps"
EPOCHS=1
# if decide to save the best model 
# should set the save and eval strategy as the same 
SAVE_BEST=False
SAVE="steps"
SAVE_STEPS=100
EVAL="steps"
EVAL_STEPS=0.1

wandb offline
#export CUDA_VISIBLE_DEVICES=0,1

# /data/llx/llama-7b-hf/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348

# /data/llx/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9

# --num_gpus $GPUS_PER_NODE --num_nodes $SLURM_NNODES

mkdir -p $OUTPUT_DIR
cp $0 $OUTPUT_DIR

deepspeed  --master_addr "localhost" --master_port $MASTER_PORT \
    --include=localhost:$CUDE_VISIBLE_DEVICES \
    /data/mxy/Finstruction/LLMzoo_mxy/train_random.py \
    --deepspeed /data/mxy/Finstruction/LLMzoo_mxy/deepspeed.conf \
    --model_name_or_path $MODEL \
    --model_max_length 2048 \
    --data_path_dir $DATA_PATH_DIR \
    --data_type  $DATA_TYPE \
    --merge_file $MERGE_FILE \
    --task_list  $TASK_LIST \
    --data_config $DATA_CONFIG \
    --output_dir $OUTPUT_DIR \
    --bf16 True \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size 8  \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy $SAVE \
    --save_steps $SAVE_STEPS \
    --load_best_model_at_end $SAVE_BEST \
    --evaluation_strategy $EVAL \
    --eval_steps $EVAL_STEPS \
    --logging_strategy "steps" \
    --logging_steps 20 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True | tee $OUTPUT_DIR/log
    
    # --data_train_path $TRAIN_PATH \
    # --data_eval_path $EVAL_PATH \


# nohup bash train.sh  >> llama2_7b_LIMA.log 2>&1 &

# (sleep 2h ; nohup bash train.sh  >> llama2_7b_lr2e-5_bs16_baseline_apaca_gpt4_total_32681_5times_std.log 2>&1 &) &

# (sleep 5s ; date) &

