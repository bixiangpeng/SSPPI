import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import coo_matrix

# class Contrast(nn.Module):
#     def __init__(self, out_dim, tau, keys):
#         super(Contrast, self).__init__()
#         self.proj = nn.ModuleDict({k: nn.Sequential(
#             nn.Linear(out_dim, out_dim),
#             nn.ELU(),
#             nn.Linear(out_dim, out_dim)
#         ) for k in keys})
#         self.tau = tau
#         for k, v in self.proj.items():
#             for model in v:
#                 if isinstance(model, nn.Linear):
#                     nn.init.xavier_normal_(model.weight, gain=1.414)
#
#     def sim(self, z1, z2):
#         z1_norm = torch.norm(z1, dim=-1, keepdim=True) #按-1维度求1范数
#         z2_norm = torch.norm(z2, dim=-1, keepdim=True)
#         dot_numerator = torch.mm(z1, z2.t())
#         dot_denominator = torch.mm(z1_norm, z2_norm.t())
#         sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
#         return sim_matrix
#
#     def compute_loss(self, str1, seq1,str2,seq2,str_memory,seq_memory, pos1,pos2):
#         z_str1 = self.proj['str'](str1)
#         z_seq1 = self.proj['seq'](seq1)
#         z_str2 = self.proj['str'](str2)
#         z_seq2 = self.proj['seq'](seq2)
#
#         z_seq_memory = self.proj['seq'](seq_memory)
#         z_str_memory = self.proj['str'](str_memory)
#
#         matrix_str2seq_memory_1 = self.sim(z_str1, z_seq_memory)
#         matrix_seq2str_memory_1 = self.sim(z_seq1, z_str_memory)
#
#         matrix_str2seq_memory_2 = self.sim(z_str2, z_seq_memory)
#         matrix_seq2str_memory_2 = self.sim(z_seq2, z_str_memory)
#
#         lori_str_1 = -torch.sum(F.log_softmax(matrix_str2seq_memory_1, dim=1) * pos1, dim=1).mean()
#         # matrix_str2seq_memory_1 = matrix_str2seq_memory_1 / (torch.sum(matrix_str2seq_memory_1, dim=1).view(-1, 1) + 1e-8)
#         # temp = matrix_str2seq_memory_1.mul(pos1).sum(dim=-1).view(-1,1)
#         # lori_str_1 = -torch.log(temp).mean()
#
#         lori_seq_1 = -torch.sum(F.log_softmax(matrix_seq2str_memory_1, dim=1) * pos1, dim=1).mean()
#         # matrix_seq2str_memory_1 = matrix_seq2str_memory_1 / (torch.sum(matrix_seq2str_memory_1, dim=1).view(-1, 1) + 1e-8)
#         # lori_seq_1 = -torch.log(matrix_seq2str_memory_1.mul(pos1).sum(dim=-1)).mean()
#
#         lori_str_2 = -torch.sum(F.log_softmax(matrix_str2seq_memory_2, dim=1) * pos2, dim=1).mean()
#         # matrix_str2seq_memory_2 = matrix_str2seq_memory_2 / (torch.sum(matrix_str2seq_memory_2, dim=1).view(-1, 1) + 1e-8)
#         # lori_str_2 = -torch.log(matrix_str2seq_memory_2.mul(pos2).sum(dim=-1)).mean()
#
#         lori_seq_2 = -torch.sum(F.log_softmax(matrix_seq2str_memory_2, dim=1) * pos2, dim=1).mean()
#         # matrix_seq2str_memory_2 = matrix_seq2str_memory_2 / (torch.sum(matrix_seq2str_memory_2, dim=1).view(-1, 1) + 1e-8)
#         # lori_seq_2 = -torch.log(matrix_seq2str_memory_2.mul(pos1).sum(dim=-1)).mean()
#
#         return lori_str_1 + lori_seq_1 + lori_str_2 + lori_seq_2
#
#     def forward(self, str1, seq1,str2,seq2,str_memory,seq_memory, pos1,pos2):
#         return self.compute_loss(str1, seq1,str2,seq2,str_memory,seq_memory, pos1,pos2)



class Contrast(nn.Module):
    def __init__(self, out_dim, tau, keys):
        super(Contrast, self).__init__()
        self.proj = nn.ModuleDict({k: nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ELU(),
            nn.Linear(out_dim, out_dim)
        ) for k in keys})
        self.tau = tau
        for k, v in self.proj.items():
            for model in v:
                if isinstance(model, nn.Linear):
                    nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True) #按-1维度求1范数
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def compute_loss(self, str1, str2, seq1, seq2, p1_index, p2_index, device):
        str = torch.cat((str1,str2), dim = 0)
        p_index = torch.cat((p1_index,p2_index), dim=0).cpu().numpy()

        seq = torch.cat((seq1, seq2), dim=0)
        z_str = self.proj['str'](str)
        z_seq = self.proj['seq'](seq)

        matrix_str2seq = self.sim(z_str, z_seq)
        matrix_seq2str = matrix_str2seq.t()

        index_ = []
        for i, index in enumerate(p_index):
            temp = np.argwhere(p_index == index)
            for j in temp:
                index_.append([i, j[0]])
        index_ = np.transpose(np.array(index_))
        src_, target_ = index_[0], index_[1]
        temp_data2 = np.ones((src_.shape[0]), dtype=int)
        pos_ = torch.LongTensor(coo_matrix((temp_data2, (src_, target_)), shape=(matrix_seq2str.shape[0], matrix_seq2str.shape[1])).toarray()).to(device)


        matrix_str2seq = matrix_str2seq / (torch.sum(matrix_str2seq, dim=1).view(-1, 1) + 1e-8)
        lori_str = -torch.log(matrix_str2seq.mul(pos_).sum(dim=-1)).mean()
        matrix_seq2str = matrix_seq2str / (torch.sum(matrix_seq2str, dim=1).view(-1, 1) + 1e-8)
        lori_seq = -torch.log(matrix_seq2str.mul(pos_).sum(dim=-1)).mean()

        # z_H_ppi = self.proj['ppi'](H_ppi)
        # z_h_ppi = torch.cat((z_H_ppi[p1_index],z_H_ppi[p2_index]),dim = 0)
        # matrix_str2ppi = self.sim(z_str, z_H_ppi) #[64,2497]
        # matrix_ppi2str = self.sim(z_h_ppi,z_str) #[64,64]
        # index_str2ppi = np.transpose(np.array([ [id, index] for id, index in enumerate(p_index)]))
        # src, target = index_str2ppi[0], index_str2ppi[1]
        # temp_data1 = np.ones((src.shape[0]), dtype=int)
        # pos_str2ppi = torch.LongTensor(coo_matrix((temp_data1, (src, target)), shape=(matrix_str2ppi.shape[0], matrix_str2ppi.shape[1])).toarray()).to(device)
        # index_ppi2str = []
        # for i, index in enumerate(p_index):
        #     temp = np.argwhere(p_index == index)
        #     for j in temp:
        #         index_ppi2str.append([i,j[0]])
        # index_ppi2str = np.transpose(np.array(index_ppi2str))
        # src_, target_ = index_ppi2str[0], index_ppi2str[1]
        # temp_data2 = np.ones((src_.shape[0]), dtype=int)
        # pos_ppi2str = torch.LongTensor(coo_matrix((temp_data2, (src_, target_)),shape=(matrix_ppi2str.shape[0], matrix_ppi2str.shape[1])).toarray()).to(device)
        # matrix_str2ppi = matrix_str2ppi / (torch.sum(matrix_str2ppi, dim=1).view(-1, 1) + 1e-8)
        # lori_str = -torch.log(matrix_str2ppi.mul(pos_str2ppi).sum(dim=-1)) .mean()
        # matrix_ppi2str = matrix_ppi2str / (torch.sum(matrix_ppi2str, dim=1).view(-1, 1) + 1e-8)
        # lori_ppi = -torch.log(matrix_ppi2str.mul(pos_ppi2str).sum(dim=-1)).mean()
        return lori_str + lori_seq

    def forward(self, str1, str2, seq1, seq2, p1_index, p2_index, device):
        return self.compute_loss(str1, str2, seq1, seq2, p1_index, p2_index, device)


# class Contrast(nn.Module):
#     def __init__(self, out_dim, tau):
#         super(Contrast, self).__init__()
#         self.proj =  nn.Sequential(
#             nn.Linear(out_dim, out_dim),
#             nn.ELU(),
#             nn.Linear(out_dim, out_dim)
#         )
#         self.tau = tau
#
#         for model in self.proj:
#             if isinstance(model, nn.Linear):
#                 nn.init.xavier_normal_(model.weight, gain=1.414)
#
#     def sim(self, z1, z2):
#         z1_norm = torch.norm(z1, dim=-1, keepdim=True) #按-1维度求1范数
#         z2_norm = torch.norm(z2, dim=-1, keepdim=True)
#         dot_numerator = torch.mm(z1, z2.t())
#         dot_denominator = torch.mm(z1_norm, z2_norm.t())
#         sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
#         return sim_matrix
#
#     def compute_loss(self, str, seq, pos):
#         z_str = self.proj(str)
#         z_seq = self.proj(seq)
#
#         matrix_str2seq = self.sim(z_str, z_seq)
#         matrix_seq2str = matrix_str2seq.t()
#
#         matrix_str2seq = matrix_str2seq / (torch.sum(matrix_str2seq, dim=1).view(-1, 1) + 1e-8)
#         lori_str = -torch.log(matrix_str2seq.mul(pos).sum(dim=-1)) .mean()
#
#         matrix_seq2str = matrix_seq2str / (torch.sum(matrix_seq2str, dim=1).view(-1, 1) + 1e-8)
#         lori_seq = -torch.log(matrix_seq2str.mul(pos).sum(dim=-1)).mean()
#         return lori_str + lori_seq
#
#     def forward(self, str, seq, pos):
#         return self.compute_loss(str, seq, pos)