import torch
import json
import argparse
import os
import numpy as np 
import tqdm
# for lan in 

class FIM_extract():
    def subnet(folder:str,threshold: str = '1',part: str = "bias"):
        _fisher = "FIM_entire.pt"
        _polar = "polar.json"

        fisher = torch.load(os.path.join(folder,_fisher))
        polar = json.load(open(os.path.join(folder,_polar)))

        p_fisher = {}
        p_mask = {}
        for k,v in fisher.items():
            if part in k:
                p_fisher[k] = np.array(v)
                p_mask[k] = np.array(v) >= polar[threshold]

        # torch.save(fisher,os.path.join(path,"mask.pt"))
        np.save(os.path.join(folder,f"{part}_mask_{threshold}"),p_mask)
        np.save(os.path.join(folder,f"{part}_fisher"),p_fisher)

def add_polar(folder):
    D = []
    FIM = torch.load(os.path.join(folder,"FIM_entire.pt"))
    polar = dict()
    bar = tqdm.tqdm(range(len(FIM.keys())))
    for k,v in FIM.items():  
        v = torch.tensor(v).flatten()
        D = np.concatenate((D,v))
        bar.update(1)
        # if "fc1.weight" in k or "fc2.weight" in k:
        #     fc_fisher[k] = np.array(v)
        #     mask[k] = np.array(v) >= polar["0.1"]
    # D =
    for threshold in range(1,10):
        p = np.percentile(np.array(D),int(threshold*10))
        polar[threshold] = p
    # import json
    # torch.save(fisher,os.path.join(path,"mask.pt"))
    # np.save(os.path.join(path,"fc_mask_1"),mask)
    # np.save(os.path.join(path,"fc_fisher_1"),fc_fisher)
    print(polar)
    json.dump(polar,open(os.path.join(folder,"polar_ex.json"),'w'))


if __name__=="__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--folder",type=str,required=True,help="Folder of FIM")
    # parser.add_argument("--threshold",type=float,required="The shreshold of fisher information mask")
    # parser.add_argument("--part",type=float,required="The portion of the model")

    # arg=parser.parse_args()

    # folder = arg.folder
    # threshold = int(arg.threshold)
    # part= arg.part
    add_polar(folder="/data/mxy/Finstruction/FIM_pool/test2")
