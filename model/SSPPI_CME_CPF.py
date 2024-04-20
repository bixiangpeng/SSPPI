# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年11月20日
"""
import torch.nn as nn
import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, GCNConv,GATConv,SuperGATConv
from contrast import Contrast
from .utils import FFN, PositionalEncoding
from .GraphTransformer import GT_block
from torch_geometric.utils import to_dense_batch


class my_MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=8, dropout=0, batch_first=True, alpha = 1, need_weights = False): #alpha = 3
        super(my_MultiHeadAttention, self).__init__()
        self.need_weights = need_weights
        self.alpha = alpha
        self.MHA = torch.nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=batch_first)
        self.ffn = FFN(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_Q, input_K, input_V, attn_mask = None, key_padding_mask=None):
        residual = input_Q
        outputs = self.MHA(input_Q, input_K, input_V, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=self.need_weights)[0]
        outputs = self.norm(outputs + self.alpha * residual)
        # outputs = outputs + residual
        outputs = self.ffn(outputs)

        return outputs


class SA_block(nn.Module):
    def __init__(self, layer = 1, hidden_dim=64, num_heads=8, dropout=0, batch_first=True):
        super(SA_block, self).__init__()
        self.pos_emb = PositionalEncoding(hidden_dim, max_len=45)
        self.SA_encoder = nn.ModuleList([ my_MultiHeadAttention(hidden_dim = hidden_dim, num_heads = num_heads, dropout = dropout, batch_first = batch_first, need_weights = False) for i in range(layer)])
    def forward(self,  inputs_Q, attn_mask, key_padding_mask ):
        outputs = self.pos_emb(inputs_Q.transpose(0, 1)).transpose(0, 1)
        for layer in self.SA_encoder:
            outputs = layer(outputs, outputs, outputs, attn_mask = attn_mask, key_padding_mask = key_padding_mask)
        return outputs


class SSPPI(nn.Module):
    def __init__(self,input_dim = 1280, hidden_dim = 64, output_dim = 1 ): #多分类的输出维度为7
        super(SSPPI, self).__init__()

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.2)

        self.tau = 0.5
        self.keys = ['str', 'seq']
        self.constrast = Contrast(hidden_dim, self.tau, self.keys)

        self.pre_transform = nn.Linear(input_dim,hidden_dim)
        self.pe_embedding = nn.Linear(20, hidden_dim)

        '''图注意力层'''
        # self.GCN1 = GATConv(hidden_dim, hidden_dim,1)
        # self.GCN2 = GATConv(hidden_dim, hidden_dim,1)
        # self.GCN3 = GATConv(hidden_dim, hidden_dim,1)

        '''SuperGAT层'''
        # self.GCN1 = SuperGATConv(hidden_dim, hidden_dim, heads = 1,dropout=0.2, attention_type='MX',edge_sample_ratio=0.8, is_undirected=True)
        # self.GCN2 = SuperGATConv(hidden_dim, hidden_dim, heads = 1,dropout=0.2, attention_type='MX',edge_sample_ratio=0.8, is_undirected=True)
        # self.GCN3 = SuperGATConv(hidden_dim, hidden_dim, heads = 1,dropout=0.2, attention_type='MX',edge_sample_ratio=0.8, is_undirected=True)

        '''图卷积层'''
        # self.GCN1 = GCNConv(hidden_dim, hidden_dim)
        # self.GCN2 = GCNConv(hidden_dim, hidden_dim)
        # self.GCN3 = GCNConv(hidden_dim, hidden_dim)
        self.GT1 = GT_block(hidden_dim, n_heads = 1)
        self.GT2 = GT_block(hidden_dim, n_heads = 1)

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size = 3)#, stride = 4)
        self.mx1 = nn.AvgPool1d(5, stride=5)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size= 3) #, stride = 4)
        self.mx2 = nn.AvgPool1d(5, stride=5)
        self.conv3 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3) #, stride = 4)
        self.mx3 = nn.AvgPool1d(45, stride=1) #2x2: 3x3:130 4x4:72 5x5:45  6x6:30 7x7:22 8x8:16 9x9:12

        self.self_attn1 = SA_block( layer = 1, hidden_dim = hidden_dim, num_heads=8, dropout=0, batch_first=True)
        self.cross_attn1 = my_MultiHeadAttention(hidden_dim = hidden_dim, num_heads = 8,dropout = 0, batch_first = True)
        self.cross_attn2 = my_MultiHeadAttention(hidden_dim =hidden_dim, num_heads= 8, dropout = 0, batch_first = True)

        self.fc1 = nn.Linear(hidden_dim * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, output_dim)


    def forward(self,  pro1 , pro2,device):

        x1,edge_index1,rwpe1,batch1, p1_index, c_size1 = pro1.x, pro1.edge_index, pro1.rwpe, pro1.batch, pro1.ppi_id,pro1.c_size
        x1 = self.pre_transform(x1)
        rwpe1 = self.pe_embedding(rwpe1)
        x1 = x1 + rwpe1


        x2, edge_index2, rwpe2,batch2, p2_index, c_size2 = pro2.x, pro2.edge_index, pro2.rwpe, pro2.batch, pro2.ppi_id, pro2.c_size
        x2 = self.pre_transform(x2)
        rwpe2 = self.pe_embedding(rwpe2)
        x2 = x2 + rwpe2

        '''GraphFormer'''
        x1, rec_loss1 = self.GT1(x1, edge_index1, c_size1, batch1, device)
        x1, rec_loss11 = self.GT2(x1, edge_index1, c_size1, batch1, device)
        x2, rec_loss2 = self.GT1(x2, edge_index2, c_size2, batch2, device)
        x2, rec_loss22 = self.GT2(x2, edge_index2, c_size2, batch2, device)
        rec_loss = None
        if self.training:
            rec_loss = rec_loss1 + rec_loss2 + rec_loss11 + rec_loss22

        '''ConvFormer'''
        seq1 = pro1.seq_data
        seq2 = pro2.seq_data
        seq1 = seq1.permute(0,2,1)
        seq1 = self.dropout(self.relu(self.conv1(seq1)))
        seq1 = self.mx1(seq1)
        seq1 = self.dropout(self.relu(self.conv2(seq1)))
        seq1 = self.mx2(seq1)
        seq1 = self.relu(self.conv3(seq1))
        seq1_res = seq1.transpose(1, 2)
        seq1_ = self.self_attn1(seq1_res, attn_mask=None, key_padding_mask=None)

        seq2 = seq2.permute(0, 2, 1)
        seq2 = self.dropout(self.relu(self.conv1(seq2)))
        seq2 = self.mx1(seq2)
        seq2 = self.dropout(self.relu(self.conv2(seq2)))
        seq2 = self.mx2(seq2)
        seq2 = self.relu(self.conv3(seq2))
        seq2_res = seq2.transpose(1, 2)
        seq2_ = self.self_attn1(seq2_res, attn_mask=None, key_padding_mask=None)

        '''Cross-modality Enhancer'''
        #图池化
        x1_c = global_mean_pool(x1, batch1)
        x2_c = global_mean_pool(x2, batch2)
        #序列池化
        seq1 = self.mx3(seq1)
        seq2 = self.mx3(seq2)





        '''Cross-protein Fusion'''

        hc = torch.cat( [seq1_to_2, seq2_to_1], dim=1 )

        hc = self.fc1(hc)
        hc = self.relu(hc)
        hc = self.dropout(hc)
        hc = self.fc2(hc)
        hc = self.relu(hc)
        hc = self.dropout(hc)
        out = self.out(hc)
        return out, torch.tensor(0, device= device), rec_loss #att_loss




