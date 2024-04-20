# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2024年01月31日
"""
import torch
import torch.nn as nn

class BilinearAttention(nn.Module):
    def __init__(self, d, num_heads):
        super(BilinearAttention, self).__init__()
        self.d = d
        self.num_heads = num_heads
        self.linear1 = nn.Linear(d, d)
        self.linear2 = nn.Linear(d, d)
        self.relu = nn.ReLU()
        self.q = nn.Parameter(torch.zeros(int(d/num_heads)))
        # self.linear3 = nn.Linear(num_heads * d, d)
        # self.linear4 = nn.Linear(num_heads * d, d)

    def forward(self, A, B):
        # A_ = self.act(self.linear1(A))
        # B_ = self.act(self.linear2(B))
        # att_maps = torch.einsum('bnd,bmd,hd->bnmh', A_, B_,self.q)
        # b2a_scores = torch.softmax(torch.mean(att_maps,2), dim=1)
        # a2b_scores = torch.softmax(torch.mean(att_maps,1), dim=1)
        # A_p = self.act(self.linear3(torch.einsum('bnd, bnh-> bdh', A, b2a_scores).view(A.shape[0],-1)))
        # B_p = self.act(self.linear4(torch.einsum('bmd, bmh-> bdh', B, a2b_scores).view(B.shape[0],-1)))
        A_ = self.relu(self.linear1(A)).view(A.shape[0], self.num_heads, A.shape[1], -1)
        B_ = self.relu(self.linear2(B)).view(B.shape[0], self.num_heads, B.shape[1], -1)
        A_ = A_.unsqueeze(3).repeat(1, 1, 1, B.shape[1], 1)
        B_ = B_.unsqueeze(2).repeat(1, 1, A.shape[1], 1, 1)
        att_maps = torch.matmul(torch.tanh(A_ *  B_),self.q)
        temp_b2a = torch.softmax(torch.mean(att_maps,3), dim=2)
        temp_a2b = torch.softmax(torch.mean(att_maps,2), dim=2)
        b2a_scores = temp_b2a.unsqueeze(-1).repeat(1,1,1,int(self.d/self.num_heads))
        a2b_scores = temp_a2b.unsqueeze(-1).repeat(1,1,1,int(self.d/self.num_heads))
        A_p = torch.sum(A.view(A.shape[0], self.num_heads, A.shape[1], -1) * b2a_scores, dim=2).view(A.shape[0],-1)
        B_p = torch.sum(B.view(B.shape[0], self.num_heads, B.shape[1], -1) * a2b_scores, dim=2).view(B.shape[0], -1)

        return torch.cat((A_p,B_p),1), torch.mean(temp_b2a,1), torch.mean(temp_a2b,1)


class BilinearAttention2(nn.Module):
    def __init__(self, d, num_heads):
        super(BilinearAttention2, self).__init__()
        self.d = d
        self.num_heads = num_heads
        self.linear1 = nn.Linear(d, d)
        self.linear2 = nn.Linear(d, d)
        self.relu = nn.ReLU()
        self.q = nn.Parameter(torch.zeros(int(d/num_heads)))

    def mutual_attention_func(self,total_AB):
        A = total_AB[0].unsqueeze(0)
        B = total_AB[1].unsqueeze(0)

        A_ = A.view(A.shape[0], self.num_heads, A.shape[1], -1)
        B_ = B.view(B.shape[0], self.num_heads, B.shape[1], -1)
        A_ = A_.unsqueeze(3).repeat(1, 1, 1, B.shape[1], 1)
        B_ = B_.unsqueeze(2).repeat(1, 1, A.shape[1], 1, 1)
        att_maps = torch.matmul(torch.tanh(A_ * B_), self.q)
        temp_b2a = torch.softmax(torch.mean(att_maps, 3), dim=2)
        temp_a2b = torch.softmax(torch.mean(att_maps, 2), dim=2)
        b2a_scores = temp_b2a.unsqueeze(-1).repeat(1, 1, 1, int(self.d / self.num_heads))
        a2b_scores = temp_a2b.unsqueeze(-1).repeat(1, 1, 1, int(self.d / self.num_heads))
        A_p = torch.sum(A.view(A.shape[0], self.num_heads, A.shape[1], -1) * b2a_scores, dim=2).view(A.shape[0], -1)
        B_p = torch.sum(B.view(B.shape[0], self.num_heads, B.shape[1], -1) * a2b_scores, dim=2).view(B.shape[0], -1)
        return A_p, B_p

    def forward(self, A , batch1, B, batch2):
        x1 = self.linear1(A)
        x2 = self.linear2(B)
        #########################################

        mark_h1 = list(torch.unique(batch1, return_counts=True)[1].cpu().tolist())
        mark_h2 = list(torch.unique(batch2, return_counts=True)[1].cpu().tolist())

        splited_h1 = torch.split(x1, mark_h1, dim=0)
        splited_h2 = torch.split(x2, mark_h2, dim=0)

        h1_total, h2_total = zip(*list(map(self.mutual_attention_func, list(zip(splited_h1, splited_h2)))))
        h1_total, h2_total = torch.vstack(list(h1_total)), torch.vstack(list(h2_total))

        return torch.cat((h1_total, h2_total),dim =1)

