MERGE_FILE=/data/mxy/Finstruction/data/train/super_ni_convert/merge.json
TASK_LIST='PrepositionPrediction'
OUTPUT=/data/mxy/Finstruction/output/test
SHARE_TASK_DiR=/data/mxy/Finstruction/data/train/super_ni_convert/fs_3
TAGET_TASK_DiR=/data/mxy/Finstruction/data/eval/gsm

mkdir -p $OUTPUT

python ./src/train/data_mix.py \
    --merge_file $MERGE_FILE \
    --task_list $TASK_LIST \
    --output $OUTPUT/data_config.json \
    --taget_task_dir $TAGET_TASK_DiR \
    --share_task_dir $SHARE_TASK_DiR