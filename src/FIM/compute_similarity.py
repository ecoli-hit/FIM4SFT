import numpy as np 
import seaborn as sns
import os
import torch
import matplotlib.pyplot as plt
import tqdm
import logging
import sys
import argparse
import os


def add_parser():
    parse = argparse.ArgumentParser()
    parse.add_argument("--folder",type=str,required=True,help="The folder contain the FIM")
    parse.add_argument("--part",type=str,required=True,help="The prefix of the FIM file")
    parse.add_argument("--outfolder",type=str,required=True,help="The output folder of the overlap similarity matrix")

    parse.add_argument("--mode",type=str,required=True,help="The output prefix of the overlap similarity matrix file") 
    parse.add_argument("--type",type=str,required=True,help="The output prefix of the overlap similarity matrix file") 
    
    parse.add_argument("--target_folder",type=str,required=True,help="The output prefix of the overlap similarity matrix file") 

    return parse.parse_args()

def overlap(i,j):
    total = 0
    overlap = 0
    for k,v in i.items() :
        overlap += ((i[k] * j[k])==True).sum()
        total += ((i[k])==True).sum()
    return overlap/total

def KL(fisher, f1, f2):
    # p1 main
    dis1 = 0
    dis2 = 0
    for k, v in fisher[f1].items():
        c1 = np.array(fisher[f1][k])
        c2 = np.array(fisher[f2][k])
        dis1 += np.sum((c2) * (np.log(c2 + 1e-16) - np.log(c1 + 1e-16)) )
        dis2 += np.sum((c1) * (np.log(c1 + 1e-16) - np.log(c2 + 1e-16)) )
        
        # dis1 += np.sum((np.exp(c1) * (c1 - c2) * (mask[f1][k] | mask[f2][k])))
        # dis2 += np.sum((np.exp(c2) * (c2 - c1) * (mask[f1][k] | mask[f2][k])))
    return dis1, dis2

def MSE(fisher, f1, f2):
    dis = 0
    num = 0
    for k, v in fisher[f1].items():
        c1 = np.array(fisher[f1][k])
        c2 = np.array(fisher[f2][k])
        distance = (c1 - c2)**2
        dis += np.sum(distance)
        num += c1.size
    mse = dis/num
    return mse,mse

def inner_similarity(args):
    folder = args.folder
    folder_list = os.listdir(folder)
    part = args.part
    mode = args.mode

    metrixs = {'KL': KL, "MSE": MSE}
    metrix = metrixs[mode]

    fisher = {}

    # load 
    bar1 = tqdm.tqdm(range(len(folder_list)))
    for s in folder_list:
        fisher[s] = np.load(os.path.join(
            folder, s, f"{part}_fisher"), allow_pickle=True).item()
        bar1.update(1)
    del bar1

    # hock = overlap
    out = np.zeros([len(folder_list),len(folder_list)])
    
    bar2 = tqdm.tqdm(range((len(folder_list)**2 + len(folder_list)//2)))
    logging.info('info beging computing')
    for index1, p1 in enumerate(folder_list):
        for index2, p2 in enumerate(folder_list):
            if p1 == p2:
                bar2.update(1)
                continue
            
            out[index1, index2], out[index2, index1] = metrix(fisher, p1, p2)
            bar2.update(1)


    # elif mode == "MSE":
    #     bar2 = tqdm.tqdm(range((len(folder_list)**2 + len(folder_list)//2)))
    #     logging.info('info beging computing')
    #     for index1, p1 in enumerate(folder_list):
    #         for index2, p2 in enumerate(folder_list):
    #             if p1 == p2:
    #                 bar2.update(1)
    #                 continue
                
    #             out[index1, index2], out[index2, index1] = MSE(fisher, p1, p2)
    #             bar2.update(1)
    

    # out = np.load("/data/mxy/Clustering/measure/fc_overlap_7.npy")
    mask = np.zeros_like(out, dtype=bool)  
    for i in range(len(folder_list)):
        mask[i,i] = True

    # save fig and data
    fig ,ax= plt.subplots(figsize=[40,40])         
    sns.heatmap(data=out,annot=True,mask=mask,cmap="Blues",cbar=False)
    ax.set_xticklabels(folder_list)
    ax.set_yticklabels(folder_list) 
    plt.savefig(f"{args.outfolder}/inner_{part}.png",dpi=300)
    np.save(f"{args.outfolder}/inner_{part}",np.array(out))

def task_aware(args):
    folder = args.folder
    folder_list = os.listdir(folder)
    part = args.part
    mode = args.mode

    metrixs = {'KL': KL, "MSE": MSE}
    metrix = metrixs[mode]

    fisher = {}

    # load 
    bar = tqdm.tqdm(range(len(folder_list)))
    for s in folder_list:
        fisher[s] = np.load(os.path.join(
            folder, s, f"{part}_fisher"), allow_pickle=True).item()
        bar.update(1)

    target = np.load(os.path.join(
            args.target_folder, f"{part}_fisher"), allow_pickle=True).item()

    bar = tqdm.tqdm(range((len(folder_list))))
    logging.info('info beging computing')
    out = []
    for index, p1 in enumerate(folder_list):
            out[index], _ = metrix(fisher, target, p1)
            bar.update(1)

    # save fig and data 
    fig ,ax= plt.subplots() 
    width = 0.4        
    plt.bar(folder_list, out, width=width)
    ax.set_xticklabels(folder_list)
    task_name = args.target_folder.split('/')[-1]
    plt.savefig(f"{args.outfolder}/task_aware_{task_name}_{part}.png",dpi=300)
    np.save(f"{args.outfolder}/task_aware_{task_name}_{part}",np.array(out))

if __name__ == "__main__":
    args = add_parser()
    if args.type == "inner":
        inner_similarity(args)
    if args.type == "task":
        task_aware(args)