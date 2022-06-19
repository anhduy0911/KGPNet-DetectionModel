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

def adj_loss_recur(pseudo_scores, adj_matrix, gt_labels):
    '''
    Calculate the auxilary loss given the pseudo scores and gtruth
    pseudo_scores - n_rois, n_class
    gtruth - no fixed number, the # of unique classes 
    '''
    # import pdb; pdb.set_trace()
    def recur_prop_ij(i, j, indx, k):
        '''
        recursively calculate joint probability of class i and class j in the image
        '''
        cur_pi = pseudo_scores[indx - 1, i]
        cur_pj = pseudo_scores[indx - 1, j]

        if indx == k * CFG.roi_per_img + 2:
            a = pseudo_scores[indx - 2, i] + (1 - pseudo_scores[indx - 2, i]) * cur_pi
            b = pseudo_scores[indx - 2, j] + (1 - pseudo_scores[indx - 2, j]) * cur_pj
            c = pseudo_scores[indx - 2, i] * (1 - cur_pj) + (1 - pseudo_scores[indx - 2, j]) * cur_pi
            d = pseudo_scores[indx - 2, j] * (1 - cur_pi) + (1 - pseudo_scores[indx - 2, i]) * cur_pj
            e = cur_pi * pseudo_scores[indx - 2, j] + pseudo_scores[indx - 2, i] * cur_pj
            return a, b, c, d, e
        else:
            a_prev, b_prev, c_prev, d_prev, e_prev = recur_prop_ij(i, j, indx - 1, k)
            a = a_prev + (1 - a_prev) * cur_pi
            b = b_prev + (1 - b_prev) * cur_pj
            c = c_prev * (1 - cur_pj) + (1 - b_prev) * (1 - a_prev) * cur_pi
            d = d_prev * (1 - cur_pi) + (1 - a_prev) * (1 - b_prev) * cur_pj
            e = e_prev + d_prev * cur_pi + c_prev * cur_pj
            return a, b, c, d, e
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
                _, _, _, _, e = recur_prop_ij(i, j, CFG.roi_per_img * (k + 1), k)
                loss += adj_matrix[i,j] * e
                denominator += adj_matrix[i,j]
        if k == 0:
            f_loss = loss / (denominator + 1e-8)
        else:
            f_loss += loss / (denominator + 1e-8)
    # print(f'adj_loss_recur: {time.time() - start_t}')
    f_loss = f_loss / len(gt_labels)
    f_loss = -torch.log(f_loss)
    return f_loss

def recur_prop_ij(pseudo_scores, i, j, indx):
    '''
    recursively calculate joint probability of class i and class j in the image
    '''
    if indx == 2:
        a = pseudo_scores[indx - 2, i] + (1 - pseudo_scores[indx - 2, i]) * pseudo_scores[indx - 1, i]
        b = pseudo_scores[indx - 2, j] + (1 - pseudo_scores[indx - 2, j]) * pseudo_scores[indx - 1, j]
        c = pseudo_scores[indx - 2, i] * (1 - pseudo_scores[indx - 1, j]) + (1 - pseudo_scores[indx - 2, j]) * pseudo_scores[indx - 1, i]
        d = pseudo_scores[indx - 2, j] * (1 - pseudo_scores[indx - 1, i]) + (1 - pseudo_scores[indx - 2, i]) * pseudo_scores[indx - 1, j]
        e = pseudo_scores[indx - 1, i] * pseudo_scores[indx, j] + pseudo_scores[indx, i] * pseudo_scores[indx - 1, j]
        return a, b, c, d, e
    else:
        print(f'{i} {j}')
        a_prev, b_prev, c_prev, d_prev, e_prev = recur_prop_ij(pseudo_scores, i, j, indx - 1)
        a = a_prev + (1 - a_prev) * pseudo_scores[indx - 1, i]
        b = b_prev + (1 - b_prev) * pseudo_scores[indx - 1, j]
        c = c_prev * (1 - pseudo_scores[indx - 1, j]) + (1 - b_prev) * pseudo_scores[indx - 1, i]
        d = d_prev * (1 - pseudo_scores[indx - 1, i]) + (1 - a_prev) * pseudo_scores[indx - 1, j]
        e = e_prev + d_prev * pseudo_scores[indx - 1, i] + c_prev * pseudo_scores[indx - 1, j]
        return a, b, c, d, e
if __name__ == '__main__':

    # for i in range(200):
        pseu = torch.softmax(torch.rand(128 * 4, 97, device='cuda:0', requires_grad=True), dim=1)
        # mask = torch.rand(96 * 128, 96 * 128)
        adj = torch.rand(97, 97, device='cuda:0')

        import time
        start_t = time.time()
        # loss = adj_based_loss(a, mask, adj, torch.tensor([1,2,3],dtype=torch.long))
        loss = adj_loss_recur(pseu, adj, [torch.tensor([82, 93, 96], device='cuda:0'), torch.tensor([51, 96], device='cuda:0')])
        # a, b, c, d, loss = recur_prop_ij(pseu, 1, 2, 256)
        print(loss)
        print(time.time() - start_t)
        assert(loss > 0)