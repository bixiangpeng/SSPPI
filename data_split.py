# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年04月05日
"""
import json
from rdkit import Chem
import numpy as np
import networkx as nx
import pickle
import pandas as pd
from collections import OrderedDict
import argparse

def data_split_for_5_fold(dataset):
    """ The train set of davis was split into 5 subsets for finetuning hyper-parameter."""
    train_pns = []
    df = pd.read_csv(f'/media/ST-18T/xiangpeng/TAGPPI-main/data/{dataset}/actions/train_cmap.actions.tsv',delimiter='\t',header = None,names = ['pro1','pro2','interaction'])
    df = df.sample(frac=1).reset_index(drop=True)
    # with open(f'/media/ST-18T/xiangpeng/TAGPPI-main/data/{args.datasetname}/actions/train_cmap.actions.tsv', 'r') as fh:
    #     for line in fh:
    #         line = line.strip('').split('\t')
    #         train_pns.append((line[0], words[1], int(words[2])))
    portion = int(0.2 * len(df['pro1']))

    for fold in range(5):
        if fold < 4:
            df_test = df.iloc[fold * portion:(fold + 1) * portion]
            df_train = pd.concat([df.iloc[:fold * portion], df.iloc[(fold + 1) * portion:]], ignore_index=True)
        if fold == 4:
            df_test = df.iloc[fold * portion:]
            df_train = df.iloc[:fold * portion]
        assert (len(df_test) + len(df_train)) == len(df)
        df_test.to_csv(f'data/{dataset}/5_fold/valid{fold + 1}.tsv', sep='\t' , index=False, header = False)
        df_train.to_csv(f'data/{dataset}/5_fold/train{fold + 1}.tsv', sep = '\t', index=False, header = False)


def main(args):
    data_split_for_5_fold(args.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default= 'yeast', help='dataset name',choices=['yeast'])
    args = parser.parse_args()
    main(args)
