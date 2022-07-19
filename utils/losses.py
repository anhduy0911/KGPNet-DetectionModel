from unittest.util import sorted_list_difference
import torch
import config as CFG
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

def adj_based_loss(pseudo_scores, masks, adj_matrix, gt_labels):
    '''
    Calculate the auxilary loss given the pseudo scores and masks
    pseudo_scores - n_rois, n_class
    masks - n_class * n_rois, n_class * n_rois
    '''
    adj_mask = torch.zeros_like(adj_matrix)
    adj_mask[gt_labels[:, None], gt_labels] = adj_matrix[gt_labels[:, None], gt_labels]

    N, C = pseudo_scores.shape
    flt_scores = pseudo_scores.flatten().unsqueeze(1).contiguous()
    flt_mat = flt_scores * flt_scores.t()
    # flt_mat = torch.log(flt_mat)
    flt_mat = flt_mat * masks
    flt_mat = flt_mat.reshape(C, N, C, N).transpose(1, 2).contiguous()
    adj_mask = adj_mask.unsqueeze(-1).unsqueeze(-1).contiguous()
    flt_mat_weighted = flt_mat * adj_mask
    loss = torch.sum(flt_mat_weighted) / torch.sum(adj_mask)
    # print(loss)
    loss = -torch.log(loss)
    return loss
    # return torch.clamp(loss, 0)

def adj_loss_recur_fs(pseudo_scores, adj_matrix, gt_rois, gt_labels):
    '''
    Calculate the auxilary loss given the pseudo scores and gtruth
    pseudo_scores - n_rois, n_class
    gtruth - no fixed number, the # of unique classes 
    '''
    # import cProfile
    # profiler = cProfile.Profile()
    # profiler.enable()
    # import pdb; pdb.set_trace()
    roi_per_img = CFG.roi_per_img
    # import pdb; pdb.set_trace()
    def loop_prop_ij(i, j, k):
        '''
        recursively calculate joint probability of class i and class j in the image
        '''
        p_mask = gt_rois[k*roi_per_img: (k+1)*roi_per_img] != 96
        indices = p_mask.nonzero().squeeze()
        p = pseudo_scores[indices, :]
        a_prev = p[0, i] + (1 - p[0, i]) * p[1, i]
        b_prev = p[0, j] + (1 - p[0, j]) * p[1, j]
        c_prev = p[0, i] * (1 - p[1, j]) + (1 - p[0, j]) * p[1, i]
        d_prev = p[0, j] * (1 - p[1, i]) + (1 - p[0, i]) * p[1, j]
        e_prev = p[1, i] * p[0, j] + p[0, i] * p[1, j]
        for k in range(2, len(p)):
            cur_pi = pseudo_scores[k, i]
            cur_pj = pseudo_scores[k, j]
            a = a_prev + (1 - a_prev) * cur_pi
            b = b_prev + (1 - b_prev) * cur_pj
            c = c_prev * (1 - cur_pj) + (1 - b_prev) * (1 - a_prev) * cur_pi
            d = d_prev * (1 - cur_pi) + (1 - a_prev) * (1 - b_prev) * cur_pj
            e = e_prev + d_prev * cur_pi + c_prev * cur_pj
            a_prev, b_prev, c_prev, d_prev, e_prev = a, b, c, d, e

        return a_prev, b_prev, c_prev, d_prev, e_prev

    f_loss = None
    for k in range(len(gt_labels)):
        loss = 0
        denominator = 0
        for i_, i in enumerate(gt_labels[k]):
            if i == 96:
                continue
            for j_, j in enumerate(gt_labels[k]):
                if j_ <= i_:
                    continue
                if j == 96:
                    continue
                _, _, _, _, e = loop_prop_ij(i, j, k)
                # print(f'{i} {j} {e}')
                loss += adj_matrix[i,j] * e
                denominator += adj_matrix[i,j]
        
        if k == 0:
            f_loss = loss / (denominator + 1e-8)
        else:
            f_loss += loss / (denominator + 1e-8)
    # print(f'adj_loss_recur: {time.time() - start_t}')
    f_loss = f_loss / len(gt_labels)

    if f_loss == 0:
        return torch.tensor(0).to(pseudo_scores.device)
    
    f_loss = -torch.log(f_loss)
    # profiler.disable()
    # stats = profiler.dump_stats('tmp.txt')
    # exit()
    return f_loss

def adj_based_loss_2(pseudo_scores, adj_matrix, roi_features, gt_labels, eps=1e-8):
    N, C = pseudo_scores.shape
    k = len(gt_labels)
    class_features = torch.mm(pseudo_scores.t(), roi_features)
    dot_inp = torch.matmul(class_features, class_features.t())
    norm_inp = torch.norm(class_features, dim=1) + eps
    norm_mtx_inp = torch.matmul(norm_inp.unsqueeze(1), norm_inp.unsqueeze(0))
    cosine_inp = dot_inp / norm_mtx_inp
    cosine_inp = 1/2 * (cosine_inp + 1)
    cosine_inp = cosine_inp - torch.eye(C).to(cosine_inp.device)
    cosine_inp = cosine_inp / torch.max(cosine_inp, dim=1)[0]

    loss = 0
    for i in range(k):
        condensed_cosine = cosine_inp[gt_labels[i]]
        condensed_adj = adj_matrix[gt_labels[i]]
        loss_parallel = torch.sum(condensed_cosine * condensed_adj) / (torch.sum(condensed_adj) + eps)
        # print(loss_parallel)
        loss += loss_parallel
    
    # print('____________________')
    loss = loss / k

    return -loss

def adj_based_loss_3(pseudo_scores, adj_matrix, gt_labels, eps=1e-8):
    k = len(gt_labels)
    sorted_scores = torch.sort(pseudo_scores, dim=0, descending=True)[0][0]
    condensed_sorted_scores = sorted_scores.unsqueeze(1)
    coocurent_scores = torch.mm(condensed_sorted_scores, condensed_sorted_scores.t())
    # import pdb; pdb.set_trace()
    loss = 0
    for i in range(k):
        condensed_adj_matrix = adj_matrix[gt_labels[i]]
        condensed_coocurent_scores = coocurent_scores[gt_labels[i]]
        loss_parallel = torch.sum(condensed_coocurent_scores * condensed_adj_matrix) / (torch.sum(condensed_adj_matrix) + eps)
        loss += loss_parallel
    loss = loss / k
    
    # import pdb; pdb.set_trace()
    return -torch.log(loss)

if __name__ == '__main__':
    # for i in range(200):
        pseu = torch.softmax(torch.rand(128 * 4, 97, device='cuda:0', requires_grad=True), dim=1)
        # mask = torch.rand(96 * 128, 96 * 128)
        adj = torch.rand(97, 97, device='cuda:0')

        gt_rois = torch.randint(0, 96, (128 * 4,), device='cuda:0')
        # import time
        # start_t = time.time()
        # # loss = adj_based_loss(a, mask, adj, torch.tensor([1,2,3],dtype=torch.long))
        # loss = adj_loss_recur_fs(pseu, adj, gt_rois, [[82, 93, 96],[51, 96]])
        # print(loss)
        # print(time.time() - start_t)
        start_time = time.time()
        # loss = adj_loss_recur(pseu, adj, [[82, 93, 96],[51, 96]])
        roi_feats =  torch.rand(128 * 4, 1024, device='cuda:0')
        # loss = adj_based_loss_2(pseu, adj, roi_feats, [torch.tensor([82, 93, 96]),torch.tensor([51, 96])])
        loss = adj_based_loss_3(pseu, adj, [torch.tensor([82, 93, 96]),torch.tensor([51, 96])])
        # a, b, c, d, loss = recur_prop_ij(pseu, 1, 2, 256)
        print(loss)
        print(time.time() - start_time)
        # assert(loss > 0)