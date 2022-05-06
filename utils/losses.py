import torch


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