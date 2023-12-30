
import json
import random
import re
import os
# import pandas as pd
import argparse
from instruction_encode_templates import encode_instruction_example, encode_few_shot_example

import tqdm
import jsonlines


def split_data(folder):
    path = folder

    with open(os.path.join(path,"data.json"), "r+", encoding="utf8") as fp:
        with jsonlines.open(os.path.join(path,"train.json"),mode='w') as writer_t:
            with jsonlines.open(os.path.join(path,"valid.json"),mode='w') as writer_v:
                cnt = 0 
                for item in jsonlines.Reader(fp):
                    if cnt%20 == 0 :
                        writer_v.write(item)
                    else:
                        writer_t.write(item)
                    cnt +=1 


def convert_super_ni_data(data_dir, output_dir, n_few_shot=2):
    os.makedirs(os.path.join(output_dir,f"fs_{n_few_shot}"), exist_ok=True)
    train_tasks = []
    with open(os.path.join(data_dir, "splits", "xlingual", "train_tasks.txt"), "r") as fin:
        for line in fin:
            if not "_mmmlu_" in line:   # skip mmlu to avoid test leakage
                train_tasks.append(line.strip())
    # with open(os.path.join(output_dir, "super_ni_data.jsonl"), "w") as fout:

    bar = tqdm.tqdm(range(len(train_tasks)))
    for task in train_tasks:
        os.makedirs(os.path.join(output_dir,f"fs_{n_few_shot}",task), exist_ok=True)
        with open(os.path.join(data_dir, "tasks", f"{task}.json"), "r") as fin:
            task_data = json.load(fin)
        
        instruction = task_data["Definition"][0]
        domains = task_data["Domains"]
        categories = task_data["Categories"]
        instances = task_data["Instances"]
        
        # for instance in instances[:zero_shot_examples_per_task]:
        #     encoded_example = encode_instruction_example(
        #         instruction=instruction, 
        #         input=instance["input"], 
        #         output=instance["output"][0],
        #         random_template=True,
        #         eos_token=None
        #     )
        #     fout.write(json.dumps({
        #         "dataset": "super_ni",
        #         "id": f"super_ni_{instance['id']}",
        #         "messages": [
        #             {"role": "user", "content": encoded_example["prompt"]},
        #             {"role": "assistant", "content": encoded_example["completion"]},
        #         ]
        #     }) + "\n")
        with open(os.path.join(output_dir,f"fs_{n_few_shot}",task,f"data.json"), "w") as fout:
            for instance in instances:
                if n_few_shot < len(task_data["Positive Examples"]):
                    examplars = random.sample(task_data["Positive Examples"], k=n_few_shot)
                else:
                    examplars = task_data["Positive Examples"]
                encoded_example = encode_few_shot_example(
                    instruction=instruction,
                    examplars=examplars,
                    input=instance["input"],
                    output=instance["output"][0],
                    eos_token=None
                )
                fout.write(json.dumps({
                    "id": f"{instance['id']}",
                    "messages": [
                        {"role": "user", "content": encoded_example["prompt"]},
                        {"role": "assistant", "content": encoded_example["completion"]},
                    ]
                }) + "\n")

        split_data(os.path.join(output_dir,f"fs_{n_few_shot}",task))

        bar.update(1)
def convert_gsm(data_dir, output_dir, n_few_shot=2):
    
    with open(os.path.join(data_dir,f"train.jsonl"), "r") as fin:
        task_data = jsonlines.Reader(fin)
        
        os.makedirs(os.path.join(output_dir,"gsm",f'instruction'), exist_ok=True)
        with open(os.path.join(output_dir,"gsm",f'instruction',"data.json"),'w') as fout:
            for idx, instance in enumerate(task_data):
                encoded_example = encode_instruction_example(
                    instruction=instance["question"],
                    input=None,
                    output=instance["answer"],
                    eos_token=None
                )
                fout.write(json.dumps({
                    "id": f"{idx}",
                    "messages": [
                        {"role": "user", "content": encoded_example["prompt"]},
                        {"role": "assistant", "content": encoded_example["completion"]},
                    ]
                }) + "\n")
    split_data(os.path.join(output_dir,"gsm",f'instruction'))

def convert_tydiqa(data_dir, output_dir, n_few_shot=2):
    
    with open(os.path.join(data_dir,f"train.json"), "r") as fin:
        task_data = json.load(fin)
        
        task_data = task_data['data']
        os.makedirs(os.path.join(output_dir,"tydiqa",f'instruction'), exist_ok=True)
        with open(os.path.join(output_dir,"tydiqa",f'instruction',"data.json"),'w') as fout:
            for idx, instance in enumerate(task_data):
                encoded_example = encode_instruction_example(
                    instruction=instance['paragraphs'][0]['qas'][0]['question'],
                    input=instance['paragraphs'][0]['context'],
                    output=instance['paragraphs'][0]['qas'][0]['answers'][0]['text'],
                    eos_token=None
                )
                fout.write(json.dumps({
                    "id": f"{idx}",
                    "messages": [
                        {"role": "user", "content": encoded_example["prompt"]},
                        {"role": "assistant", "content": encoded_example["completion"]},
                    ]
                }) + "\n")
    split_data(os.path.join(output_dir,"tydiqa",f'instruction'))

def convert_tqa(data_dir, output_dir, n_few_shot=2):
    
    with open(os.path.join(data_dir,f"data.json"), "r") as fin:
        task_data = jsonlines.Reader(fin)
        
        os.makedirs(os.path.join(output_dir,"truthfulqa",f'instruction'), exist_ok=True)
        with open(os.path.join(output_dir,"truthfulqa",f'instruction',"data.json"),'w') as fout:
            for idx, instance in enumerate(task_data):
                encoded_example = encode_instruction_example(
                    instruction=instance["prompt"],
                    input=None,
                    output=instance["completion"],
                    eos_token=None
                )
                fout.write(json.dumps({
                    "id": f"{idx}",
                    "messages": [
                        {"role": "user", "content": encoded_example["prompt"]},
                        {"role": "assistant", "content": encoded_example["completion"]},
                    ]
                }) + "\n")
    split_data(os.path.join(output_dir,"truthfulqa",f'instruction'))

if __name__=='__main__':
    # convert_super_ni_data("/data/mxy/Finstruction/data/raw_train/natural-instructions-2.8","/data/mxy/Finstruction/data/train/super_ni_convert",3)
    convert_tqa("/data/mxy/Finstruction/data/eval/truthfulqa",
                "/data/mxy/Finstruction/data/train")