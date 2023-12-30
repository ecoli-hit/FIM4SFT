import copy
import json
import logging
from dataclasses import dataclass
from typing import Dict, Sequence
import os
import torch
import transformers
from torch.utils.data import Dataset
import logging
import pathlib
from typing import Dict, Optional, Sequence,List
import torch 
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from llmzoo.constants import IGNORE_INDEX, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN
from llmzoo.utils import default_conversation
import jsonlines



def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:

    data_type = data_args.data_type
    SUPPORT_DATASETS={'super_ni': InstructionDatasetofSni,
                      'single': InstructionDatasetofSni}
    assert data_type in SUPPORT_DATASETS.keys()

    dataset_cls = SUPPORT_DATASETS[data_type]
    
    
    if data_type == "single":
        train_dataset = dataset_cls(tokenizer=tokenizer, data_path=os.path.join(data_args.data_path_dir,'train.json'))
        eval_dataset = dataset_cls(tokenizer=tokenizer,  data_path=os.path.join(data_args.data_path_dir,'valid.json'))
    
    if data_type == "super_ni":
        # 实现多个数据集直接concat，只需要输入路径
        data_path_dir=data_args.data_path_dir
        merge_file = data_args.merge_file
        task_list = data_args.task_list.split(',')
        
        folder_list = []
        with open(merge_file,'r') as fp:
            fin = json.load(fp=fp)
            for task in task_list:
                for f in fin[task]["files"]:
                    folder_list.append(os.path.join(data_path_dir,f))

        if len(folder_list) >1 :
            datasets_train=[]
            datasets_eval=[]
            for data_path in folder_list:
                print(f"==== load data from {data_path} ====")
                datasets_train.append(dataset_cls(tokenizer=tokenizer,
                            data_path=os.path.join(data_path,'train.json')))
                datasets_eval.append(dataset_cls(tokenizer=tokenizer,
                            data_path=os.path.join(data_path,'valid.json')))

            train_dataset=ConcatDataset(datasets_train)
            eval_dataset=ConcatDataset(datasets_eval)
            print(f"datasets cumulative_sizes: {train_dataset.cumulative_sizes}")
        else:
            train_dataset = dataset_cls(tokenizer=tokenizer, data_path=os.path.join(folder_list[0],'train.json'))
            eval_dataset = dataset_cls(tokenizer=tokenizer,  data_path=os.path.join(folder_list[0],'valid.json'))
    
    

    if data_type == "hybrid":
        with open(data_args.data_config,'r') as fin:
            data_config = json.load(fin)
        
        datasets_train=[]
        datasets_eval=[]
        for folder in data_config['auxiliary']:
            print(f"==== load data from {folder} ====")
            datasets_train.append(dataset_cls(tokenizer=tokenizer,
                        data_path=os.path.join(folder,'train.json')))
            
            
        datasets_eval.append(dataset_cls(tokenizer=tokenizer,
                    data_path=os.path.join(folder,'valid.json')))
            
        train_dataset=ConcatDataset(datasets_train)
        eval_dataset=ConcatDataset(datasets_eval)
        print(f"datasets cumulative_sizes: {train_dataset.cumulative_sizes}")
            
    # else:
    #     if data_args.data_path_dir is None:
    #         train_dataset = dataset_cls(tokenizer=tokenizer, data_path=data_args.data_train_path)
    #         eval_dataset = dataset_cls(tokenizer=tokenizer, data_path=data_args.data_eval_path)
    #         # train_dataset = dataset_cls(tokenizer=tokenizer, data_path=data_args.data_path)

    #     else:
    #         # 实现多个数据集直接concat，只需要输入路径
    #         data_path_dir=data_args.data_path_dir
    #         # import glob
    #         # file_list = glob.glob(os.path.join(data_path_dir,"*.json"))
    #         # file_list = sorted(file_list)
    #         folder_list = []
    #         for i,j,k in os.walk(data_path_dir):
    #             if k == []:
    #                 continue
    #             folder_list.append(i)
    #         datasets_train=[]
    #         datasets_eval=[]
    #         for data_path in folder_list:
    #             print(data_path)
    #             datasets_train.append(dataset_cls(tokenizer=tokenizer,
    #                         data_path=os.path.join(data_path,'train.jsonl')))
    #             datasets_eval.append(dataset_cls(tokenizer=tokenizer,
    #                         data_path=os.path.join(data_path,'valid.jsonl')))

    #         train_dataset=ConcatDataset(datasets_train)
    #         eval_dataset=ConcatDataset(datasets_eval)
    #         print(f"datasets cumulative_sizes: {train_dataset.cumulative_sizes}")
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


# class InstructionDataset(Dataset):
#     def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, ):
#         super(InstructionDataset, self).__init__()
#         logging.info("Loading data...")
#         list_data_dict = json.load(open(data_path, "r"))
#         list_data_dict = _prepro_data_dict(list_data_dict)
#         self.tokenizer = tokenizer
#         self.list_data_dict = list_data_dict

#     def __len__(self):
#         return len(self.list_data_dict)

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         sources = self.list_data_dict[i]
#         if isinstance(i, int):
#             sources = [sources]
#         data_dict = preprocess(copy.deepcopy([e["conversations"] for e in sources]), self.tokenizer)
#         if isinstance(i, int):
#             data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
#         return data_dict


class InstructionDatasetofCode(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, ):
        super(InstructionDatasetofCode, self).__init__()
        logging.info("Loading data...")
        list_data_dict = load_dataset_code(data_path)
        list_data_dict = _prepro_data_dict(list_data_dict)
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        data_dict = preprocess(copy.deepcopy([e["conversations"] for e in sources]), self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        return data_dict

def load_dataset_code(data_path: str):
    print(f"load data from {data_path}")
    data_dict_list = []
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            human = {'from':'human','value': item['subject'] + '\n'+ item['old_contents'] }
            gpt = {'from':'gpt','value': item["new_contents"]}
            data_dict_list.append({'conversations': [human , gpt]})
    return data_dict_list

class InstructionDatasetofSni(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, ):
        super(InstructionDatasetofSni, self).__init__()
        logging.info("Loading data...")
        list_data_dict = load_dataset_sni(data_path)
        list_data_dict = _prepro_data_dict(list_data_dict)
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        data_dict = preprocess(copy.deepcopy([e["conversations"] for e in sources]), self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        return data_dict

def load_dataset_sni(data_path: str):
    print(f"load data from {data_path}")
    data_dict_list = []
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, line in enumerate(jsonlines.Reader(f)):
            item = line["messages"]
            human = {'from':'human','value': item[0]['content']}
            gpt = {'from':'gpt','value': item[1]["content"]}
            data_dict_list.append({'conversations': [human , gpt]})
    return data_dict_list



@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    intermediates = []
    for source in sources:
        header = f"{default_conversation.system}"
        conversation, intermediate = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
        intermediates.append(intermediate)

    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)

    # keep only machine responses as targets
    assert len(targets) == len(intermediates)
    for target, inters in zip(targets, intermediates):
        mask = torch.zeros_like(target, dtype=torch.bool)
        for inter in inters:
            tokenized = _tokenize_fn(inter, tokenizer)
            start_idx = tokenized["input_ids"][0].size(0) - 1
            end_idx = tokenized["input_ids"][1].size(0)
            mask[start_idx:end_idx] = True
        target[~mask] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=targets)


def _add_speaker_and_signal(header, source, get_conversation=True):
    BEGIN_SIGNAL = DEFAULT_BOS_TOKEN
    END_SIGNAL = DEFAULT_EOS_TOKEN
    conversation = ""
    intermediate = []
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = default_conversation.roles[1]
        else:
            from_str = 'unknown'
        # store the string w/o and w/ the response
        value = (BEGIN_SIGNAL + sentence["value"] + END_SIGNAL)
        if sentence["from"].lower() == "gpt":
            start = conversation + BEGIN_SIGNAL
            end = conversation + value
            intermediate.append([start, end])
        if get_conversation:
            conversation += value
    return conversation, intermediate


def _prepro_data_dict(list_data_dict):
    list_data_dict = [item for item in list_data_dict if len(item["conversations"]) > 0]
    return list_data_dict


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

if __name__ == '__main__':
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained("/data/mxy/models/llama2-7b-hf")
    path = "/data/mxy/Finstruction/data/code/commitpackft/data/abap/data.jsonl"
    dataset = make_supervised_data_module(path,tokenizer=tokenizer)
    
    lenth = dataset.__len__()
    for i in range(lenth):
        dataclass.__getitem__(i)