import json
import os
import tqdm

if __name__=='__main__':
    train_tasks = []
    cats = dict()
    data_dir = "/data/mxy/Finstruction/data/raw_train/natural-instructions-2.8"
    output_dir = "/data/mxy/Finstruction/data/train/super_ni_convert"
    with open(os.path.join(data_dir, 
                           "splits", "xlingual", "train_tasks.txt"), "r") as fin:
        for line in fin:
            if not "_mmmlu_" in line:   # skip mmlu to avoid test leakage
                train_tasks.append(line.strip())
    
    bar = tqdm.tqdm(range(len(train_tasks)))
    for task in train_tasks:
        with open(os.path.join(data_dir, "tasks", f"{task}.json"), "r") as fin:
            task_data = json.load(fin)
        
        
        instances = len(task_data["Instances"])
        categorie = task_data["Categories"][0].replace(' ', '_')

        if categorie not in cats.keys():
            cats[categorie] = {"files":[task], "instances": instances}
        else:
            cats[categorie]["files"].append(task)
            cats[categorie]["instances"] += instances
        bar.update(1)
    print(cats.keys())
    json.dump(cats,open(os.path.join(output_dir,"merge.json"),'w'))