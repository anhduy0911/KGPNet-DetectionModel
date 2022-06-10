import torch
import config as CFG
import os

def create_loss_mask(n_rois, n_class=CFG.n_classes, device='cuda'):
    '''
    Create loss mask for auxarilary loss.
    '''
    flt_shape = n_rois * n_class
    mask = torch.ones((flt_shape, flt_shape), dtype=torch.float32, device=device)
    for i in range(flt_shape):
        for j in range(flt_shape):
            if i // n_rois == j // n_rois:
                mask[i, j] = 0
            elif (j-i) % n_rois == 0:
                mask[i, j] = 0

    torch.save(mask, os.path.join(CFG.base_log, 'mask.pth'))
    return mask

if __name__ == '__main__':
    mask = create_loss_mask(128, 96, device='cuda')
    import matplotlib.pyplot as plt
    import seaborn

    print(mask)
    seaborn.heatmap(mask[:25, :25])
    plt.savefig('mask.png')