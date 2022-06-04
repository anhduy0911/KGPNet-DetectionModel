import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import math
from torch_geometric.utils import remove_self_loops
import torch_sparse
from torch_scatter import scatter_add

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_size) -> None:
        super(GCN, self).__init__()
        # nn_baseblock = nn.Sequential(
        #     nn.Linear(input_dim, hidden_size*2),
        #     nn.Linear(hidden_size*2, hidden_size*2),
        #     nn.BatchNorm1d(hidden_size*2),
        #     nn.LeakyReLU()
        # )

        # nn_baseblock2 = nn.Sequential(
        #     nn.Linear(hidden_size*2, hidden_size),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.BatchNorm1d(hidden_size),
        #     nn.LeakyReLU()
        # )
        # self.conv1 = pyg_nn.GINConv(nn_baseblock)
        self.conv1 = pyg_nn.GATv2Conv(input_dim, hidden_size * 2, edge_dim=1)
        self.tanh = nn.Tanh()
        # self.conv2 = pyg_nn.GCNConv(32, 32)
        # self.conv3 = pyg_nn.GCNConv(32, 32)
        # self.conv4 = pyg_nn.GCNConv(32, 32)
        # self.conv5 = pyg_nn.GINConv(nn_baseblock2)
        self.conv5 = pyg_nn.GATv2Conv(hidden_size * 2, hidden_size, edge_dim=1)
    
    def forward(self, x, edge_idx, edge_w):
        # print(x.shape)
        x = self.conv1(x, edge_idx, edge_w)
        x = self.tanh(x)
        # x = self.conv2(x, edge_idx, edge_w)
        # x = self.relu(x)
        # x = self.conv3(x, edge_idx, edge_w)
        # x = self.relu(x)
        # x = self.conv4(x, edge_idx, edge_w)
        # x = self.relu(x)
        x = self.conv5(x, edge_idx, edge_w)
        return x

class GTN(nn.Module):
    '''
    Graph Transformer Network
    '''
    def __init__(self, num_edge, num_channels, w_in, w_out, num_nodes, num_layers):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.num_nodes = num_nodes
        self.w_in = w_in
        self.w_out = w_out
        self.num_layers = num_layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, num_nodes, first=True))
            else:
                layers.append(GTLayer(num_edge, num_channels, num_nodes, first=False))
        self.layers = nn.ModuleList(layers)
        # self.loss = nn.CrossEntropyLoss()
        self.gcn = pyg_nn.GCNConv(in_channels=self.w_in, out_channels=w_out)
        self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out)
        self.linear2 = nn.Linear(self.w_out, self.w_out)

    def normalization(self, H):
        norm_H = []
        for i in range(self.num_channels):
            edge, value=H[i]
            edge, value = remove_self_loops(edge, value)
            deg_row, deg_col = self.norm(edge, self.num_nodes, value)
            value = deg_col * value
            norm_H.append((edge, value))
        return norm_H

    def norm(self, edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                    dtype=dtype,
                                    device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)
        row, col = edge_index
        deg = scatter_add(edge_weight.clone(), col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row], deg_inv_sqrt[col]

    def forward(self, A, X):
        # Ws = []
        for i in range(self.num_layers):
            if i == 0:
                # H, W = self.layers[i](A)
                H = self.layers[i](A)
            else:                
                # H, W = self.layers[i](A, H)
                H = self.layers[i](A, H)
            H = self.normalization(H)
            # Ws.append(W)
        for i in range(self.num_channels):
            if i==0:
                edge_index, edge_weight = H[i][0], H[i][1]
                X_ = self.gcn(X,edge_index=edge_index, edge_weight=edge_weight)
                X_ = F.relu(X_)
            else:
                edge_index, edge_weight = H[i][0], H[i][1]
                X_ = torch.cat((X_,F.relu(self.gcn(X,edge_index=edge_index, edge_weight=edge_weight))), dim=1)
        X_ = self.linear1(X_)
        X_ = F.relu(X_)
        y = self.linear2(X_)

        return y

class GTLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_nodes, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        self.num_nodes = num_nodes
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels, num_nodes)
            self.conv2 = GTConv(in_channels, out_channels, num_nodes)
        else:
            self.conv1 = GTConv(in_channels, out_channels, num_nodes)
    
    def forward(self, A, H_=None):
        if self.first == True:
            result_A = self.conv1(A)
            result_B = self.conv2(A)                
            # W = [(F.softmax(self.conv1.weight, dim=1)),(F.softmax(self.conv2.weight, dim=1))]
        else:
            result_A = H_
            result_B = self.conv1(A)
            # W = [(F.softmax(self.conv1.weight, dim=1))]
        H = []
        for i in range(len(result_A)):
            a_edge, a_value = result_A[i]
            b_edge, b_value = result_B[i]
            
            # edges, values = torch_sparse_old.spspmm(a_edge, a_value, b_edge, b_value, self.num_nodes, self.num_nodes, self.num_nodes)
            edges, values = torch_sparse.spspmm(a_edge, a_value, b_edge, b_value, self.num_nodes, self.num_nodes, self.num_nodes)
            H.append((edges, values))
        return H

class GTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_nodes):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels))
        self.bias = None
        self.num_nodes = num_nodes
        self.reset_parameters()
    def reset_parameters(self):
        n = self.in_channels
        nn.init.normal_(self.weight, std=0.01)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        filter = F.softmax(self.weight, dim=1)
        num_channels = filter.shape[0]
        results = []
        for i in range(num_channels):
            for j, (edge_index,edge_value) in enumerate(A):
                if j == 0:
                    total_edge_index = edge_index
                    total_edge_value = edge_value*filter[i][j]
                else:
                    total_edge_index = torch.cat((total_edge_index, edge_index), dim=1)
                    total_edge_value = torch.cat((total_edge_value, edge_value*filter[i][j]))
            index, value = torch_sparse.coalesce(total_edge_index, total_edge_value, m=self.num_nodes, n=self.num_nodes)
            results.append((index, value))
        return results