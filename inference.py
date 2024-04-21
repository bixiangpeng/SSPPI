# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年07月25日
"""
import pickle as pkl
import argparse
from my_dataset import collate_pyg, PygDataset
from torch.utils.data import DataLoader
from model.SSPPI import SSPPI
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.metrics import confusion_matrix
import re
from train_and_test import test

def main(args):
    with open(f'./data/{args.datasetname}/index_map_dict.pkl', 'rb') as file:
        index_map_dict = pkl.load(file)
    if args.datasetname == 'multi_species':
        test_name = f'test_{args.identity}'
    else:
        test_name = 'test'
    test_pns = []
    with open(f'./data/{args.datasetname}/dataset/{test_name}.tsv', 'r') as fh:
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')
            words = re.split('  |\t', line)
            test_pns.append((words[0], words[1], int(words[2])))
    test_dataset = PygDataset(dataset_name=args.datasetname, pns=test_pns, map_dict=index_map_dict)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pyg,num_workers=args.num_workers)
    device = torch.device('cuda:' + str(args.device_id) if torch.cuda.is_available() else "cpu")
    model = SSPPI(output_dim=args.output_dim)
    model = model.to(device)
    if args.datasetname == 'multi_species':
        cp_name = f'model_{args.identity}'
    else:
        cp_name = f'model'
    path = f'./model_pkl/{args.datasetname}/{cp_name}.pkl'
    model.load_state_dict(torch.load(path))
    G, P_value, P_label = test(model, device, test_loader, args)

    if args.datasetname == 'multi_class':
        test_acc = accuracy_score(G, P_label)
        test_prec = precision_score(G, P_label, average='weighted')
        test_recall = recall_score(G, P_label, average='weighted')
        test_f1 = f1_score(G, P_label, average='weighted')
        test_auc = roc_auc_score(G, P_value, average='weighted', multi_class='ovr')
    else:
        test_acc = accuracy_score(G, P_label)
        test_prec = precision_score(G, P_label)
        test_recall = recall_score(G, P_label)
        test_f1 = f1_score(G, P_label)
        test_auc = roc_auc_score(G, P_value)
    con_matrix = confusion_matrix(G, P_label)
    test_spec = con_matrix[0][0] / ( con_matrix[0][0] + con_matrix[0][1] )
    test_mcc = ( con_matrix[0][0] * con_matrix[1][1] - con_matrix[0][1] * con_matrix[1][0] ) / (((con_matrix[1][1] +con_matrix[0][1]) * (con_matrix[1][1] +con_matrix[1][0]) * (con_matrix[0][0] +con_matrix[0][1]) * (con_matrix[0][0] +con_matrix[1][0])) ** 0.5)
    print("acc: ", test_acc, " ; prec: ", test_prec, " ; recall: ", test_recall, " ; f1: ", test_f1, " ; auc: ", test_auc, " ; spec:", test_spec, " ; mcc: ", test_mcc)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--datasetname', type = str , default = 'multi_species', choices=['yeast', 'multi_species', 'multi_class'])
    parse.add_argument('--output_dim', type=int, default = 1, help='1 for yeast and multi_species, while 7 for multi_class')
    parse.add_argument('--identity', type=str, default='01', choices=['01', '10', '25', '40', 'any'], help='for multi_speices')
    parse.add_argument('--device_id', type = int, default = 1)
    parse.add_argument('--batch_size', type=int, default = 32)
    parse.add_argument('--num_workers', type=int, default= 8)
    args = parse.parse_args()
    main(args)