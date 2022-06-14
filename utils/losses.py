import torch
import config as CFG

def KL_loss_fast_compute(target, input, eps=1e-6):
    '''
    Custom Approximation of KL given N samples of target dist and input dist
    target - N, C
    input - N, C
    '''
    dot_inp = torch.matmul(input, input.t())
    norm_inp = torch.norm(input, dim=1) + eps
    norm_mtx_inp = torch.matmul(norm_inp.unsqueeze(1), norm_inp.unsqueeze(0))
    cosine_inp = dot_inp / norm_mtx_inp
    cosine_inp = 1/2 * (cosine_inp + 1)
    cosine_inp = cosine_inp / torch.sum(cosine_inp, dim=1)
    
    dot_tar = torch.matmul(target, target.t())
    norm_tar = torch.norm(target, dim=1) + eps
    norm_mtx_tar = torch.matmul(norm_tar.unsqueeze(1), norm_tar.unsqueeze(0))
    cosine_tar = dot_tar / norm_mtx_tar
    cosine_tar = 1/2 * (cosine_tar + 1)
    cosine_tar = cosine_tar / torch.sum(cosine_tar, dim=1)
    
    losses = cosine_tar * torch.log(cosine_tar / cosine_inp)
    loss = torch.sum(losses)
    
    return loss

def JS_loss_fast_compute(target, input, eps=1e-6):
    '''
    Custom Approximation of KL given N samples of target dist and input dist
    target - N, C
    input - N, C
    '''
    dot_inp = torch.matmul(input, input.t())
    norm_inp = torch.norm(input, dim=1) + eps
    norm_mtx_inp = torch.matmul(norm_inp.unsqueeze(1), norm_inp.unsqueeze(0))
    cosine_inp = dot_inp / norm_mtx_inp
    cosine_inp = 1/2 * (cosine_inp + 1)
    cosine_inp = cosine_inp / torch.sum(cosine_inp, dim=1)
    
    dot_tar = torch.matmul(target, target.t())
    norm_tar = torch.norm(target, dim=1) + eps
    norm_mtx_tar = torch.matmul(norm_tar.unsqueeze(1), norm_tar.unsqueeze(0))
    cosine_tar = dot_tar / norm_mtx_tar
    cosine_tar = 1/2 * (cosine_tar + 1)
    cosine_tar = cosine_tar / torch.sum(cosine_tar, dim=1)
    
    losses_tar_inp = cosine_tar * torch.log(cosine_tar / cosine_inp)
    losses_inp_tar = cosine_inp * torch.log(cosine_inp / cosine_tar)
    loss = torch.sum(losses_tar_inp) + torch.sum(losses_inp_tar)
    
    return loss

def graph_embedding_loss(out_features, adj_matrix, threshold=0, eps=1e-6):
    '''
    Calculate the correlation between embedding vectors with its corresponding original adj matrix
    out_features - n_class, hidden_size
    adj_matrix - n_class, n_class
    '''
    N, N = adj_matrix.shape

    dot_inp = torch.matmul(out_features, out_features.t())
    norm_inp = torch.norm(out_features, dim=1) + eps
    norm_mtx_inp = torch.matmul(norm_inp.unsqueeze(1), norm_inp.unsqueeze(0))
    cosine_inp = dot_inp / norm_mtx_inp
    cosine_inp = 1/2 * (cosine_inp + 1)
    cosine_inp = cosine_inp - torch.eye(N).to(cosine_inp.device)
    cosine_inp = cosine_inp / torch.max(cosine_inp, dim=1)[0]

    mask_matrix = adj_matrix > threshold 
    adj_matrix = adj_matrix / (torch.max(adj_matrix, dim=1)[0] + eps)
    adj_matrix = (adj_matrix + adj_matrix.t()) / 2

    loss = torch.sum(torch.abs(adj_matrix - cosine_inp) * mask_matrix)    
    return loss

def adj_based_loss(pseudo_scores, masks, adj_matrix):
    '''
    Calculate the auxilary loss given the pseudo scores and masks
    pseudo_scores - n_rois, n_class
    masks - n_class * n_rois, n_class * n_rois
    '''
    vals, indices = torch.topk(adj_matrix, k=CFG.topk_neighbor, dim=-1)
        # print(topk)
    adj_matrix.zero_()
    adj_matrix[torch.arange(adj_matrix.size(0))[:, None], indices] = vals

    N, C = pseudo_scores.shape
    flt_scores = pseudo_scores.flatten().unsqueeze(1).contiguous()
    flt_mat = flt_scores * flt_scores.t()
    # flt_mat = torch.log(flt_mat)
    flt_mat = flt_mat * masks
    flt_mat = flt_mat.reshape(C, N, C, N).transpose(1, 2).contiguous()
    adj_matrix = adj_matrix.unsqueeze(-1).unsqueeze(-1).contiguous()
    flt_mat_weighted = flt_mat * adj_matrix
    loss = torch.mean(flt_mat_weighted)
    loss = -torch.log(loss)
    # import pdb; pdb.set_trace()
    return loss
    # return torch.clamp(loss, 0)

if __name__ == '__main__':

    # for i in range(200):
    a = torch.rand(128, 96)
    mask = torch.rand(96 * 128, 96 * 128)
    adj = torch.rand(96, 96)

    import time
    start_t = time.time()
    loss = adj_based_loss(a, mask, adj)
    print(loss)
    print(time.time() - start_t)