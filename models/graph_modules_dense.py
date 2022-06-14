import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import math

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
    def __init__(self, num_edge, num_channels, w_in, w_out, num_layers):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_layers = num_layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, first=True))
            else:
                layers.append(GTLayer(num_edge, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        # self.loss = nn.CrossEntropyLoss()
        self.gcn = pyg_nn.GCNConv(in_channels=self.w_in, out_channels=w_out)
        self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out)
        self.linear2 = nn.Linear(self.w_out, self.w_out)

    def normalization(self, H):
        '''
        H - dense matrix of shape K_, N, N
        '''
        norm_H = []
        for i in range(self.num_channels):
            # H_i_sparse = H[i].to_sparse()
            # edge, value= H_i_sparse.indices(), H_i_sparse.values()
            # edge, value = remove_self_loops(edge, value)
            # deg_row, deg_col = self.norm(edge, n_node, value)
            # self.norm_dense(H[i])
            # import pdb; pdb.set_trace()
            # value = deg_col * value

            # H_i_sparse.values = value
            # norm_H_dense.append(H_i_sparse.to_dense())
            # norm_H.append((edge, value))
            H_i = H[i]
            H_i.fill_diagonal_(0) # remove self-loops
            deg_nodes = torch.sum(H_i, dim=0)
            deg_inv_sqrt = deg_nodes.pow(-1)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            H_i_norm = H_i.clone() * deg_inv_sqrt.unsqueeze(1)
            norm_H.append(H_i_norm)
            # import pdb; pdb.set_trace()
        norm_H_dense = torch.stack(norm_H, dim=0)
        return norm_H_dense

    def forward(self, A, X):
        # Ws = []
        for i in range(self.num_layers):
            if i == 0:
                # H, W = self.layers[i](A)
                H = self.layers[i](A)
            else:                
                # H, W = self.layers[i](A, H)
                H = self.layers[i](A, H)
            # H = self.normalization(H)
            H = self.normalization(H)
            # Ws.append(W)
        for i in range(self.num_channels):
            H_i_sp = H[i].to_sparse()
            if i==0:
                edge_index, edge_weight = H_i_sp.indices(),  H_i_sp.values()
                X_ = self.gcn(X,edge_index=edge_index, edge_weight=edge_weight)
                X_ = F.relu(X_)
            else:
                edge_index, edge_weight = H_i_sp.indices(),  H_i_sp.values()
                X_ = torch.cat((X_,F.relu(self.gcn(X,edge_index=edge_index, edge_weight=edge_weight))), dim=1)
        X_ = self.linear1(X_)
        X_ = F.relu(X_)
        y = self.linear2(X_)

        return y

class GTLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)
        # self.conv1.weight.register_hook(lambda x: print('grad accumulated in weight'))
    def forward(self, A, H_=None):
        # print(self.conv1.weight.grad)
        if self.first == True:
            result_A = self.conv1(A) 
            result_B = self.conv2(A)                
            # W = [(F.softmax(self.conv1.weight, dim=1)),(F.softmax(self.conv2.weight, dim=1))]
        else:
            result_A = H_
            result_B = self.conv1(A)
            # W = [(F.softmax(self.conv1.weight, dim=1))]
        # H = []

        H = torch.bmm(result_A, result_B) # K_, N, N
        # import pdb; pdb.set_trace()
        # print(H)
        # for i in range(len(result_A)):
        #     a_edge, a_value = result_A[i]
        #     b_edge, b_value = result_B[i]
            
        #     print(f'a value: {a_value}')
        #     print(f'b value: {b_value}')
        #     # edges, values = torch_sparse_old.spspmm(a_edge, a_value, b_edge, b_value, self.num_nodes, self.num_nodes, self.num_nodes)
        #     torch.mm
        #     edges, values = torch_sparse.spmm(a_edge, a_value, b_edge, b_value, self.num_nodes, self.num_nodes, self.num_nodes)
        #     print(values)
        #     H.append((edges, values))
        
        return H

class GTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels))
        self.bias = None
        self.reset_parameters()
    def reset_parameters(self):
        n = self.in_channels
        nn.init.normal_(self.weight, std=0.01)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        '''
        A - dense matrices of multiple adj matrices - K, N, N
        '''
        _, N, N = A.shape
        filter = F.softmax(self.weight, dim=1) # K_, K
        # print(filter)
        retransformed = torch.matmul(filter, A.flatten(start_dim=1)) # K_, N*N
        retransformed = retransformed.view(-1, N, N) # K_, N, N
        
            # print(value)
        return retransformed