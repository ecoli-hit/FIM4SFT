for lan in 'go' 'c++' 'python' 'java' 'jsx' 'rust';do
    bash src/train_dp.sh ./data/code/commitpackft/data/$lan/data.jsonl $lan 1
    # bash /data/mxy/Finstruction/src/train_dp.sh ./data/code/commitpackft/data/$1/train.jsonl ./data/code/commitpackft/data/$1/train.jsonl  $lan'_ft_20' 20
done
