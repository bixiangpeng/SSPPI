import torch.nn as nn
import torch
import math
import numpy as np

class FFN(nn.Module):
    def __init__(self, hidden_dim = 64, alpha = 1):
        super(FFN, self).__init__()
        self.alpha = alpha
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False)
        )
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.norm(output + self.alpha * residual)
        # return output + residual
        # return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_model = 64, n_heads = 8):
        self.d_model = d_model
        self.n_heads = n_heads
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, edge_encoding = None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_model) #scores:[batch_size, n_heads, len_q, len_k]
        if edge_encoding is not None:
            scores = scores + edge_encoding.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9) #Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)

        context = torch.matmul(attn, V) #[batch_size, n_heads, len_q, d_v]
        return context, attn, scores


class MSA(nn.Module):
    def __init__(self, hidden_dim = 64, n_heads = 8):
        super(MSA, self).__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert self.head_dim * n_heads == hidden_dim, "embed_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim

        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ScaledDotProductAttention = ScaledDotProductAttention(d_model = self.head_dim, n_heads = n_heads)
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask, edge_encoding = None):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2) #Q:[bs, heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2) #K:[bs, heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2) #V:[bs, heads, len_v(=len_k), d_v]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context, attn, scores = self.ScaledDotProductAttention(Q, K, V, attn_mask, edge_encoding)
        scores = scores.sum(dim=1) / self.n_heads
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.head_dim) #context:[bs, len_q, heads*d_v]
        output = self.fc(context)
        return output, scores

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
        self.pos_emb = PositionalEncoding(hidden_dim, max_len=48)
        self.SA_encoder = nn.ModuleList([ my_MultiHeadAttention(hidden_dim = hidden_dim, num_heads = num_heads, dropout = dropout, batch_first = batch_first, need_weights = False) for i in range(layer)])
    def forward(self,  inputs_Q, attn_mask, key_padding_mask ):
        outputs = self.pos_emb(inputs_Q.transpose(0, 1)).transpose(0, 1)
        for layer in self.SA_encoder:
            outputs = layer(outputs, outputs, outputs, attn_mask = attn_mask, key_padding_mask = key_padding_mask)
        return outputs

class BilinearAttention(nn.Module):
    def __init__(self, d, num_heads):
        super(BilinearAttention, self).__init__()
        self.d = d
        self.num_heads = num_heads
        self.linear1 = nn.Linear(d, d)
        self.linear2 = nn.Linear(d, d)
        self.relu = nn.ReLU()
        self.q = nn.Parameter(torch.zeros(int(d/num_heads)))

    def forward(self, A, B):
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