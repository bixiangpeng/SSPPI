# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年03月26日
"""
import pickle as pkl
import re
from my_dataset import collate_pyg,PygDataset
from torch.utils.data import DataLoader
from model.SSPPI import SSPPI
import torch
from train_and_test import train,test
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.metrics import confusion_matrix
from torch_geometric.loader import DataLoader
import argparse


def main(args):
    with open(f'./data/yeast/index_map_dict.pkl', 'rb') as file:
        index_map_dict = pkl.load(file)
    for fold in range(1,6):
        train_pns = []
        with open(f'./data/yeast/5_fold/train{fold}.tsv','r') as fh:
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip('\n')
                words = re.split('  |\t', line)
                train_pns.append((words[0], words[1], int(words[2])))
        test_pns = []
        with open(f'./data/yeast/5_fold/valid{fold}.tsv', 'r') as fh:
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip('\n')
                words = re.split('  |\t', line)
                test_pns.append((words[0], words[1], int(words[2])))
        train_dataset = PygDataset(dataset_name=args.datasetname, pns=train_pns, map_dict=index_map_dict)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=collate_pyg, num_workers=args.num_workers)
        test_dataset = PygDataset(dataset_name=args.datasetname, pns=test_pns, map_dict=index_map_dict)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pyg, num_workers=args.num_workers)

        device = torch.device('cuda:' + str(args.device_id) if torch.cuda.is_available() else "cpu")
        model = SSPPI()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        model_file = f'/media/ST-18T/xiangpeng/SSPPI/model_pkl/test{fold}/'
        result_file = f'/media/ST-18T/xiangpeng/SSPPI/model_pkl/test{fold}/result.txt'

        for epoch in range(args.epochs):
            print("Running EPOCH", epoch + 1)
            avg_loss, acc = train(model, train_loader, device, optimizer, criterion, args)
            if (args.do_Save):
                torch.save(model.state_dict(), model_file + 'epoch' + f'{epoch}.pkl')
            G, P_value, P_label = test(model, device, test_loader, args)
            test_acc = accuracy_score(G, P_label)
            test_prec = precision_score(G, P_label)
            test_recall = recall_score(G, P_label)
            test_f1 = f1_score(G, P_label)
            test_auc = roc_auc_score(G, P_value)
            con_matrix = confusion_matrix(G, P_label)
            test_spec = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
            test_mcc = (con_matrix[0][0] * con_matrix[1][1] - con_matrix[0][1] * con_matrix[1][0]) / (((con_matrix[1][1] +con_matrix[0][1]) * (con_matrix[1][1] +con_matrix[1][0]) * (con_matrix[0][0] +con_matrix[0][1]) * (con_matrix[0][0] +con_matrix[1][0])) ** 0.5)
            print("acc: ", test_acc, " ; prec: ", test_prec, " ; recall: ", test_recall, " ; f1: ", test_f1, " ; auc: ",test_auc, " ; spec:", test_spec, " ; mcc: ", test_mcc, 'lr:', optimizer.param_groups[0]['lr'])

            with open(result_file, 'a+') as fp:
                fp.write('epoch:' + str(epoch + 1) + '\ttrainacc=' + str(acc) + '\ttrainloss=' + str(avg_loss.item()) + '\tacc=' + str(test_acc) + '\tprec=' + str(test_prec) + '\trecall=' + str(
                    test_recall) + '\tf1=' + str(test_f1) + '\tauc=' + str(test_auc) + '\tspec=' + str(
                    test_spec) + '\tmcc=' + str(test_mcc) + '\tlr=' + str(optimizer.param_groups[0]['lr']) + '\n')
        print('Fold-{} valid finished.'.format(str(fold)))

    print("5-fold cross validation finished.")


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--datasetname', type=str, default='yeast')
    parse.add_argument('--device_id', type=int, default = 0)
    parse.add_argument('--batch_size', type=int, default = 32)
    parse.add_argument('--epochs', type=int, default = 200)
    parse.add_argument('--lr', type=float, default = 0.001)
    parse.add_argument('--num_workers', type=int, default = 8)
    parse.add_argument('--do_Save', type=bool, default = True)
    parse.add_argument("--do_Test", type=bool, default = True)
    args = parse.parse_args()

    main(args)






