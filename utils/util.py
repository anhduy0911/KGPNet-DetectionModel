import torch
import config as CFG
import os
import json

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

def generate_prescription_test_dataset(pres_id):
    pill_pres_dict = json.load(open(os.path.join(CFG.detection_root, 'pill_pres_map.json'), 'r'))
    instance_test_total = json.load(open(os.path.join(CFG.pill_root, 'data_test/instances_test.json'), 'r'))
    instance_test_specific = {}

    img_list = [x[:-5] for x in pill_pres_dict[pres_id]]

    instance_test_specific['images'] = [x for x in instance_test_total['images'] if x['id'] in img_list]
    instance_test_specific['annotations'] = [x for x in instance_test_total['annotations'] if x['image_id'] in img_list]
    instance_test_specific['categories'] = instance_test_total['categories']
    
    json.dump(instance_test_specific, open(os.path.join(CFG.pill_root, 'data_test/instances_test_pres.json'), 'w'))
    return instance_test_specific

def size_investigate():
    instance_test_total = json.load(open(os.path.join(CFG.pill_root, 'data_test/instances_test.json'), 'r'))
    cnt = {'sm': 0, 'md': 0, 'lg': 0}
    for ann in instance_test_total['annotations']:
        if ann['area'] < 32**2:
            cnt['sm'] += 1
        elif ann['area'] < 96**2:
            cnt['md'] += 1
        else:
            cnt['lg'] += 1
    
    print(cnt)
    

if __name__ == '__main__':
    # mask = create_loss_mask(128, 96, device='cuda')
    # import matplotlib.pyplot as plt
    # import seaborn

    # print(mask)
    # seaborn.heatmap(mask[:25, :25])
    # plt.savefig('mask.png')
    # instances = generate_prescription_test_dataset('20220105_150733276396')
    # print(instances['images'])
    size_investigate()