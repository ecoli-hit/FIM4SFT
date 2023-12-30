import json
import argparse
import os

def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge_file",type=str,required=True)
    parser.add_argument("--task_list",type=str,required=True)
    parser.add_argument("--output",type=str,required=True)
    parser.add_argument("--taget_task_dir",type=str,required=True)
    parser.add_argument("--share_task_dir",type=str,required=True)

    return parser.parse_args()

def hybrid_datasets(args):
    data_path_dir=args.share_task_dir
    merge_file  = args.merge_file
    task_list   = args.task_list.split(',')
     
    folder_list = []
    with open(merge_file,'r') as fp:
        fin = json.load(fp=fp)
        for task in task_list:
            for f in fin[task]["files"]:
                folder_list.append(os.path.join(data_path_dir,f))
    
    with open(args.output,'w') as fout:
        fout.write(json.dumps({'auxiliary': folder_list, 'target': args.taget_task_dir}))


if __name__=='__main__':
    args = add_parser()
    hybrid_datasets(args)