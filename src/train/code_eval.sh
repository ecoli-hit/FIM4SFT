accelerate launch --gpu_ids=0 /data/mxy/Finstruction/data/code/evaluation/bigcode-evaluation-harness/main.py \
  --model /data/lyj/hf_models/llama-2-7b-hf \
  --max_length_generation 2048 \
  --prompt instruct \
  --tasks humanevalfixtests-python \
  --temperature 0.2 \
  --n_samples 20 \
  --batch_size 10 \
  --allow_code_execution \
  --generation_only \
  --save_generations_path /data/mxy/Finstruction/test_script/generations_humanevalfixpython_octocoder.json \
  --metric_output_path /data/mxy/Finstruction/test_script/evaluation_humanevalfixpython_octocoder.json \
