import torch.nn as nn
from .utils import SA_block

class Convormer(nn.Module):
    def __init__(self, input_dim = 1280, hidden_dim = 64,kernel_size = [3, 5 , 7], patch_size = 25):
        super(Convormer,self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size = kernel_size[0], padding = (kernel_size[0] - 1) // 2)
        self.patch = nn.AvgPool1d(patch_size, stride=patch_size)
        self.self_attn = SA_block(layer = 1, hidden_dim = hidden_dim, num_heads=8, dropout = 0.2, batch_first=True) #########drop

    def forward(self, pro):
        seq = pro.seq_data
        seq = seq.permute(0, 2, 1)
        seq = self.dropout(self.relu(self.conv1(seq)))
        seq = self.patch(seq)
        seq = seq.transpose(1,2)
        seq = self.self_attn(seq, attn_mask=None, key_padding_mask=None) # relu
        return  seq

class Convormer2(nn.Module):
    def __init__(self, input_dim = 1280, hidden_dim = 64,kernel_size = [3, 5 , 7], patch_size = 25):
        super(Convormer2,self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size = kernel_size[0], padding = (kernel_size[0] - 1) // 2)
        self.patch = nn.AvgPool1d(patch_size, stride=patch_size)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size[0],padding=(kernel_size[0] - 1) // 2)
        self.conv3 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size[1],padding=(kernel_size[1] - 1) // 2)
        self.conv4 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size[2],padding=(kernel_size[2] - 1) // 2)


    def forward(self, pro):
        seq = pro.seq_data
        seq = seq.permute(0, 2, 1)
        seq = self.dropout(self.relu(self.conv1(seq)))
        seq = self.patch(seq)
        seq = self.dropout(self.relu(self.conv2(seq)))
        seq = self.dropout(self.relu(self.conv3(seq)))
        seq = self.dropout(self.relu(self.conv4(seq)))
        seq = seq.transpose(1,2)

        return  seq