import torch
import time

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

def adj_based_loss_4(pseudo_scores, adj_matrix, margin=0.05):
    '''
    triplet loss computation for neighbor pills - positive couplets, non-neighbor pills - negative couplets
    '''
    # k = len(gt_labels)
    # import pdb; pdb.set_trace()
    N, C = pseudo_scores.shape

    pred_scores, pred_labels = torch.sort(pseudo_scores, dim=-1, descending=True) # N
    pred_scores = pred_scores[:, 0]
    # print(pred_scores.shape)
    pred_labels = pred_labels[:, 0]

    _, labels_neighbors =  torch.sort(adj_matrix[pred_labels], dim=-1, descending=True) # N, C
    # positive couplets for all predicted labels

    pos_labels_neighbors = labels_neighbors[:, :10] # N, 10
    # negative couplets for all predicted labels
    neg_labels_neighbors = labels_neighbors[:, -10:] # N, 10
 
    pos_scores = 1 - torch.stack([torch.index_select(pseudo_scores, dim=1, index=pos_labels_neighbors[i]) for i in range(N)], dim=0) # N, N, 10
    neg_scores = 1 - torch.stack([torch.index_select(pseudo_scores, dim=1, index=neg_labels_neighbors[i]) for i in range(N)], dim=0) # N, N, 10

    pos_scores = 1 - torch.prod(pos_scores, dim=1) # N, 10
    neg_scores = 1 - torch.prod(neg_scores, dim=1) # N, 10

    pos_scores = torch.sum(pos_scores, dim=1) # N
    neg_scores = torch.sum(neg_scores, dim=1) # N

    loss = pred_scores * neg_scores - pred_scores * pos_scores
    loss = torch.sum(loss) + margin
    # print(f'here {loss}')
    return torch.max(loss, torch.zeros(1, device=loss.device))[0]

def adj_based_loss_5(pseudo_scores, adj_matrix, gtruth_class, margin=0.05):
    '''
    triplet loss computation for neighbor pills - positive couplets, non-neighbor pills - negative couplets
    '''
    # k = len(gt_labels)
    # import pdb; pdb.set_trace()
    N, C = pseudo_scores.shape
    
    pred_scores =  pseudo_scores[torch.arange(0, N), gtruth_class]
    pred_scores_masked = pred_scores * (gtruth_class != 96)
    _, labels_neighbors =  torch.sort(adj_matrix[gtruth_class], dim=-1, descending=True) # N, C
    # positive couplets for all predicted labels

    pos_labels_neighbors = labels_neighbors[:, :3] # N, 10
    # negative couplets for all predicted labels
    neg_labels_neighbors = labels_neighbors[:, -3:] # N, 10
 
    pos_scores = 1 - torch.stack([torch.index_select(pseudo_scores, dim=1, index=pos_labels_neighbors[i]) for i in range(N)], dim=0) # N, N, 10
    neg_scores = 1 - torch.stack([torch.index_select(pseudo_scores, dim=1, index=neg_labels_neighbors[i]) for i in range(N)], dim=0) # N, N, 10

    pos_scores = 1 - torch.prod(pos_scores, dim=1) # N, 10
    neg_scores = 1 - torch.prod(neg_scores, dim=1) # N, 10

    pos_scores = torch.sum(pos_scores, dim=1) # N
    neg_scores = torch.sum(neg_scores, dim=1) # N

    loss = pred_scores_masked * neg_scores - pred_scores_masked * pos_scores
    loss = torch.sum(loss) + margin
    # import pdb; pdb.set_trace()
    return torch.max(loss, torch.zeros(1, device=loss.device))[0]

if __name__ == '__main__':
    # for i in range(200):
        pseu = torch.softmax(torch.rand(128 * 4, 97, requires_grad=True), dim=1)
        # mask = torch.rand(96 * 128, 96 * 128)
        adj = torch.rand(97, 97)

        gt_rois = torch.randint(0, 96, (4,128))
        # import time
        # start_t = time.time()x
        # # loss = adj_based_loss(a, mask, adj, torch.tensor([1,2,3],dtype=torch.long))
        # loss = adj_loss_recur_fs(pseu, adj, gt_rois, [[82, 93, 96],[51, 96]])
        # print(loss)
        # print(time.time() - start_t)
        start_time = time.time()
        # loss = adj_loss_recur(pseu, adj, [[82, 93, 96],[51, 96]])
        roi_feats =  torch.rand(128 * 4, 1024)
        # loss = adj_based_loss_2(pseu, adj, roi_feats, [torch.tensor([82, 93, 96]),torch.tensor([51, 96])])
        loss = adj_based_loss_4(pseu, adj)
        # a, b, c, d, loss = recur_prop_ij(pseu, 1, 2, 256)
        print(loss)
        print(time.time() - start_time)
        # assert(loss > 0)