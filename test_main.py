# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年07月25日
"""
import pickle as pkl
import argparse

from test_dataset import collate_pyg, PygDataset
from torch.utils.data import DataLoader
from model.visual_SSPPI import SSPPI
import torch
from train_and_test import train,test
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.metrics import confusion_matrix
import math
import matplotlib.pyplot as plt
from sklearn import manifold
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale

def visual(feat):
    # t-SNE的最终结果的降维与可视化
    ts = manifold.TSNE(n_components=2, init='random', random_state=0)
    x_ts = ts.fit_transform(feat)
    print(x_ts.shape)  # [num, 2]
    x_min, x_max = x_ts.min(0), x_ts.max(0)
    x_final = (x_ts - x_min) / (x_max - x_min)
    return x_final

def plotlabels(S_lowDWeights, Trure_labels, name):
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    print(S_data)
    print(S_data.shape)  # [num, 3]

    for index in range(7):  # 假设总共有三个类别，类别的表示为0,1,2
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=100, marker=maker[index], c=colors[index], edgecolors=colors[index], alpha=0.65)

        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值

    plt.title(name, fontsize=32, fontweight='normal', pad=20)


# 设置散点形状
maker = ['.', '.', '.', '.', '.', '.', '.']
# 设置散点颜色
colors = ['slateblue', 'crimson','red','blue','orange','gray','black']
# 图例名称
Label_Com = ['a', 'b', 'c', 'd']
# 设置字体格式
font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 32,
         }

def main(args):
    # pdb_id = '6snv' #################################
    # with open(f'/media/ST-18T/xiangpeng/SSPPI/PDB/new_input_data/{pdb_id}/graph_dict.pkl','rb') as file:
    #     pro_graph_dict = pkl.load(file)
    #
    # with open(f'/media/ST-18T/xiangpeng/SSPPI/PDB/new_input_data/{pdb_id}/seq_dict.pkl','rb') as file:
    #     seq_esm_dict = pkl.load(file)
    #
    # test_pns = [('6snv_A', '6snv_B', 1)]  #############################

    # with open(f'/media/ST-18T/xiangpeng/SSPPI/data/{args.datasetname}/graph_dict_8_PE_with_2_order.pkl', 'rb') as file:
    #     pro_graph_dict = pkl.load(file)
    # with open(f'/media/ST-18T/xiangpeng/SSPPI/data/{args.datasetname}/seq_esm_tensor_dict.pkl', 'rb') as file:
    #     seq_esm_dict = pkl.load(file)

    pro_graph_dict = None
    seq_esm_dict = None

    # test_pns = []
    # with open(f'/media/ST-18T/xiangpeng/SSPPI/data/{args.datasetname}/actions/test_cmap.actions.tsv',
    #           'r') as fh:
    #     for line in fh:
    #         line = line.strip('\n')
    #         line = line.rstrip('\n')
    #         words = re.split('  |\t', line)
    #         test_pns.append((words[0], words[1], int(words[2])))

    test_pns = []
    with open(f'/media/ST-18T/xiangpeng/SSPPI/data/{args.datasetname}/actions/test-multi-any.tsv','r') as fh:
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')
            words = re.split('  |\t', line)
            test_pns.append((words[0], words[1], int(words[2])))
    # test_pns = []
    # with open(f'/media/ST-18T/xiangpeng/SSPPI/data/{args.datasetname}/actions/test_cmap.actions.tsv', 'r') as fh:
    #     for line in fh:
    #         line = line.strip('\n')
    #         line = line.rstrip('\n')
    #         words = line.split(' ')
    #         test_pns.append((words[0], words[1], int(words[2])))

    test_dataset = PygDataset( datasetname = args.datasetname, pns=test_pns,graph_dict = pro_graph_dict, seq_dict = seq_esm_dict)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle= False, collate_fn = collate_pyg, num_workers = args.num_workers )

    device = torch.device('cuda:'+ str(args.device_id) if torch.cuda.is_available() else "cpu")
    model = SSPPI()
    model = model.to(device)
    path = '/media/ST-18T/xiangpeng/SSPPI/model_pkl/new_MS/any/any_2/epoch45.pkl' #29, 199, 372
    # path = '/media/ST-18T/xiangpeng/SSPPI/model_pkl/yeast/new_architecture/new_GT_MHMA2/epoch338.pkl'
    # path = '/media/ST-18T/xiangpeng/SSPPI/model_pkl/MC/1/epoch46.pkl'
    model.load_state_dict(torch.load(path))

    ### test
    model.eval()
    total_pred_values = torch.Tensor()
    total_pred_labels = torch.Tensor()
    total_true_labels = torch.Tensor()
    total_emb = torch.Tensor()
    # total_scores = {}
    print('Make prediction for {} samples...'.format(len(test_loader.dataset)))
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader)):
            # print(batch_idx)
            temp = {}
            pro_data1 = data[0].to(device)
            pro_data2 = data[1].to(device)
            # output, emb, scores = model(pro_data1, pro_data2, device)
            output, a_scores, b_scores, emb = model(pro_data1, pro_data2, device)
            a_scores = minmax_scale(a_scores.squeeze(0).cpu().numpy())
            b_scores = minmax_scale(b_scores.squeeze(0).cpu().numpy())
            # temp['seq1_sa'] = minmax_scale(np.sum(scores[0].squeeze(0).cpu().numpy(), axis = 0))
            # temp['seq2_sa'] = minmax_scale(np.sum(scores[1].squeeze(0).cpu().numpy(), axis = 0))
            # temp['seq1_ca1'] = minmax_scale(np.sum(scores[2].squeeze(0).cpu().numpy(), axis = 0))
            # temp['seq2_ca1'] = minmax_scale(np.sum(scores[3].squeeze(0).cpu().numpy(), axis = 0))
            # temp['seq1_ca2'] = minmax_scale(np.sum(scores[4].squeeze(0).cpu().numpy(), axis = 0))
            # temp['seq2_ca2'] = minmax_scale(np.sum(scores[5].squeeze(0).cpu().numpy(), axis = 0))
            # temp['str1_out1'] = scores[6].squeeze(0).cpu().numpy()
            # temp['str1_out2'] = minmax_scale(np.sum(scores[7].squeeze(0).cpu().numpy(), axis = 0))
            # temp['str2_out1'] = scores[8].squeeze(0).cpu().numpy()
            # temp['str2_out2'] = minmax_scale(np.sum(scores[9].squeeze(0).cpu().numpy(), axis = 0))
            # # = (pro_data1.ppi_id, scores)
            # # = (pro_data2.ppi_id, scores)
            if args.datasetname == 'multi_class':
                predicted_values = torch.softmax(output, dim=1)
                predicted_labels = torch.argmax(predicted_values, dim=1)
            else:
                predicted_values = torch.sigmoid(output)
            # # total_scores[batch_idx] = (predicted_values, data[0].y, data[0].ppi_id, data[1].ppi_id, temp)
            # all_scores = (predicted_values.cpu(), data[0].y.cpu(), temp)
            # with open(f'/media/ST-18T/xiangpeng/SSPPI/PDB/{pdb_id}_scores.pkl', 'wb') as file:
            #     pkl.dump(all_scores, file)

            predicted_labels = torch.round(predicted_values)
            total_pred_values = torch.cat((total_pred_values, predicted_values.cpu()), 0)  # predicted values
            total_pred_labels = torch.cat((total_pred_labels, predicted_labels.cpu()), 0)  # predicted labels
            total_true_labels = torch.cat((total_true_labels, pro_data1.y.view(-1, 1).cpu()), 0)  # ground truth
            total_emb = torch.cat((total_emb, emb.cpu()), dim=0)




    G, P_value, P_label, emb = total_true_labels.numpy().flatten(), total_pred_values.numpy().flatten(), total_pred_labels.numpy().flatten(), total_emb.numpy()


    ###T-SNE降维可视化
    plt.figure(figsize=(10, 10))
    v_res = visual(emb).tolist()
    G_list = G.tolist()
    emb_x = [ v[0] for v in v_res ]
    emb_y = [ v[1] for v in v_res ]
    predicted_data = {
        'emb_x': emb_x,
        'emb_y': emb_y,
        'label': G_list,
    }
    df_pre = pd.DataFrame(predicted_data)
    df_pre.to_csv(f'./results/MS/new_any_2_tsne_epoch45.csv')
    # df_pre.to_csv(f'./results/MC/1_tsne_epoch46.csv',header = None)
    #
    #
    #
    # plotlabels(v_res, G, '(a)')
    # plt.show()



    ###指标计算
    test_acc = accuracy_score(G,P_label)
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
    parse.add_argument('--datasetname', type = str , default = 'multi_species')
    parse.add_argument('--device_id', type = int, default = 1)
    parse.add_argument('--batch_size', type=int, default = 64)
    parse.add_argument('--epochs', type=int, default = 400)
    parse.add_argument('--num_workers', type=int, default= 0)
    args = parse.parse_args()

    main(args)