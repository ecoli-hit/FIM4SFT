import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from peft import get_peft_model_state_dict

from transformers import Trainer
from torch.utils.data import SequentialSampler,DistributedSampler

from typing import List, Iterator, Optional
import torch.nn as nn
import torch
import math
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data.sampler import BatchSampler, Sampler, SubsetRandomSampler, RandomSampler

from llmzoo.datasets.datasets import make_supervised_data_module
from llmzoo.models import build_model
from llmzoo.utils import safe_save_model_for_hf_trainer

import os 
import json
import numpy as np

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    lora: Optional[bool] = field(default=False)
    lora_r: Optional[int] = field(default=16)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.05)


@dataclass
class DataArguments:
    data_train_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_eval_path: str = field(default=None, metadata={"help": "Path to the eval data."})
    # data_path: str = field(default=None, metadata={"help": "Path to the eval data."})
    task_list: str  = field(default=None, metadata={"help": "A list of task, used with data_type=super_ni"})
    merge_file: str = field(default=None, metadata={"help": "Path of merge file, used with data_type=super_ni"})
    data_type: str  = field(default=None, metadata={"help": "The loaded data type"})
    data_path_dir: str=field(default=None)
    data_config: str=field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # FIM args
    dump_fim =True if training_args.num_train_epochs == 1 else False
    print(f"Dump Fisher {dump_fim}")
    dump_file = training_args.output_dir
    dump_threshould = -1

    model, tokenizer = build_model(model_args, training_args)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    global FIM
    trainer = Trainer(model=model, 
                      dump_fim=dump_fim, 
                      dump_file=dump_file,
                      dump_thrshould=dump_threshould,
                      tokenizer=tokenizer, args=training_args, **data_module)

    if model_args.lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))
        if torch.__version__ >= "2":
            model = torch.compile(model)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    ## get fisher information matrix 
    # if dump_fim:
    #     print("==== Dump Fisher Information Matrix ===")
    #     FIM = trainer._get_FIM()

    #     # D = []
    #     # polar = dict()
    #     # for k,v in FIM.items():  
    #     #     v = torch.tensor(v).flatten()
    #     #     D = np.concatenate((D,v))
    #     #     # if "fc1.weight" in k or "fc2.weight" in k:
    #     #     #     fc_fisher[k] = np.array(v)
    #     #     #     mask[k] = np.array(v) >= polar["0.1"]
    #     # # D =
    #     # for threshold in range(1,10):
    #     #     p = np.percentile(np.array(D),NotImplemented(threshold*10))
    #     #     polar[threshold] = p
    #     # # import json
    #     # # torch.save(fisher,os.path.join(path,"mask.pt"))
    #     # # np.save(os.path.join(path,"fc_mask_1"),mask)
    #     # # np.save(os.path.join(path,"fc_fisher_1"),fc_fisher)
    #     # json.dump(polar,open(os.path.join(training_args.output_dir,"polar_ex.json"),'w'))

    #     torch.save(FIM,dump_file)

    if not dump_fim:
        trainer.save_state()
        if model_args.lora:
            model.save_pretrained(os.path.join(training_args.output_dir, "lora"))
            tokenizer.save_pretrained(os.path.join(training_args.output_dir, "lora"))
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
