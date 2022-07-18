from cmath import nan
import pickle
import json
import math
import pandas as pd
import re
import config as CFG
import torch

from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse

def build_KG_graph(json_file, exclude_path='', name='pill_data'):
    '''
    build bipartite coocurence graph from extracted prescription json file
    '''
    coocurence = {}
    pill_occurence = {}
    diag_occurence = {}
    
    def convert_KG_data(json_file):
        with open(json_file) as f:
            data = json.load(f)
        for pres in data:
            for pill in pres['pills']:
                pill = pill['name']
                pill_occurence[pill] = pill_occurence.get(pill, 0) + 1
                for diag in pres['diagnose']:
                    if re.match(r'.*TD\b', diag) is None:
                        diag_code = diag.strip('()') # not end with td => exclude ()
                    else:
                        print(diag)
                        diag_code = diag # end with td => keep ()
                    diag_occurence[diag_code] = diag_occurence.get(diag_code, 0) + 1
                    if coocurence.get(pill) is None:
                        coocurence[pill] = {}
                    if coocurence.get(diag_code) is None:
                        coocurence[diag_code] = {}
                    coocurence[pill][diag_code] = coocurence[pill].get(diag_code, 0) + 1
                    coocurence[diag_code][pill] = coocurence[diag_code].get(pill, 0) + 1

    convert_KG_data(json_file)

    def tf_idf(pill, diag):
        tf = coocurence[pill][diag] / diag_occurence[diag]
        idf =  math.log( sum(diag_occurence.values()) / sum(coocurence[pill].values()))

        return tf * idf

    weighted_edges = {}

    exclude_names = []
    # if exclude_path != '':
    #     exclude_ids = pickle.load(open(exclude_path, 'rb'))
    #     name2idx = pickle.load(open('./data/pills/name2id.pkl', 'r'))
    #     exclude_names = [list(name2idx.keys())[list(name2idx.values()).index(i)] for i in exclude_ids]
    
    for pill in pill_occurence.keys():
        for diag in coocurence[pill].keys():
            # print(f'pill: {pill} diag: {diag}')
            if weighted_edges.get(pill) is None:
                weighted_edges[pill] = {}
            weighted_edges[pill][diag] = tf_idf(pill, diag)

    # print(weighted_edges)
    with open('data/graph/' + name + '.csv', 'w') as f:
        for pill in weighted_edges.keys():
            for diag, weight in weighted_edges[pill].items():
                # print('im here')
                if pill in exclude_names:
                    print(f'Excluding {pill}')
                    continue
                f.write(str(pill) + ',' + diag + ',' + str(weight) + '\n')

def build_graph_data():
    edge_index = []
    edge_weight = []
    
    pill_edge = pd.read_csv(CFG.graph_root + 'pill_pill_graph.csv', header=0)
    for x, y, w in pill_edge.values:
        edge_index.append([x, y])
        edge_weight.append(w)
        edge_index.append([y, x])
        edge_weight.append(w)
    
    data = Data(x=torch.eye(CFG.n_classes, dtype=torch.float32), edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(), edge_attr=torch.tensor(edge_weight).unsqueeze(1))
    # print(data)
    adj_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_attr).squeeze()
    # pad 0 to the end of the adj matrix
    adj_mat = torch.cat([adj_mat, torch.zeros((1, CFG.n_classes), dtype=torch.float32)], dim=0)
    adj_mat = torch.cat([adj_mat, torch.zeros((CFG.n_classes + 1, 1), dtype=torch.float32)], dim=1)
    return data, adj_mat

def build_size_graph_data():
    adj_mat = torch.zeros((CFG.n_classes, CFG.n_classes), dtype=torch.float32)
    ratio_dict = json.load(open(CFG.graph_root + 'size_ratios.json', 'r'))   
    for i in range(CFG.n_classes):
        if ratio_dict[str(i)] == 0:
            continue
        for j in range(CFG.n_classes):
            if i == j or ratio_dict[str(j)] == 0:
                continue
            adj_mat[i, j] = ratio_dict[str(j)] / ratio_dict[str(i)]
    
    adj_mat = torch.cat([adj_mat, torch.zeros((1, CFG.n_classes), dtype=torch.float32)], dim=0)
    adj_mat = torch.cat([adj_mat, torch.zeros((CFG.n_classes + 1, 1), dtype=torch.float32)], dim=1)
    return adj_mat

def merge_multilabel_meta(root, train):
    # merge multilabel meta
    if train:
        path = root + 'data_train_ai4vn/instances_train_ai4vn.json'
    else:
        path = root + 'data_test_ai4vn/instances_test_ai4vn.json'

    meta = json.load(open(path, 'r'))

    meta_dict = {}
    for att in meta['annotations']:
        pic = att['image_id']
        if pic not in meta_dict:
            meta_dict[pic] = []
        meta_dict[pic].append(att['category_id'])

    if train:
        json.dump(meta_dict, open(root + 'data_train_ai4vn/multilabel_meta.json', 'w'))
    else:
        json.dump(meta_dict, open(root + 'data_test_ai4vn/multilabel_meta.json', 'w'))

def build_size_graph(train):
    if train:
        path = 'data/pills/data_train_ai4vn/instances_train_ai4vn.json'
        path_multilabel = 'data/pills/data_train_ai4vn/multilabel_meta.json'
    else:
        path = 'data/pills/data_test/instances_test_ai4vn.json'
        
    annots = json.load(open(path, 'r'))['annotations']
    multilabel_dict = json.load(open(path_multilabel, 'r'))
    imgs_list = list(multilabel_dict.keys())
    labels_list = list(multilabel_dict.values())
    
    ratios = {}
    # isFirst = False
    co_graph, adj_matrix = build_graph_data()
    # adj_matrix_filtered = adj_matrix > torch.quantile(co_graph.edge_attr, q=0.25)
    adj_matrix_filtered = adj_matrix > 0
    
    def find_neighbor_ratio(node_idx, ratio_i):
        neighbors = (adj_matrix_filtered[node_idx] == True).nonzero(as_tuple=True)[0]
        neighbors = neighbors.tolist()
        # print(neighbors)
        for neighbor in neighbors:
            if neighbor in ratios:          
                continue
            ls_occ = [i for i, x in enumerate(labels_list) if node_idx in x and neighbor in x]
            if len(ls_occ) == 0:
                continue
            img_name = imgs_list[ls_occ[0]]
            # print(f'{node_idx}, {neighbor}, {img_name}')
            area_i = next(x['area'] for x in annots if x['image_id'] == img_name and x['category_id'] == node_idx)
            area_p = next(x['area'] for x in annots if x['image_id'] == img_name and x['category_id'] == neighbor)
            
            ratio_p = area_p / area_i * ratio_i
            ratios[neighbor] = ratio_p
            find_neighbor_ratio(neighbor, ratio_p)
    ratios[0] = 1
    find_neighbor_ratio(0, 1)
    
    # fill all remaining nodes with ratio 0
    for i in range(CFG.n_classes):
        if i not in ratios:
            print(i)
            neighbors = (adj_matrix_filtered[i] == True).nonzero(as_tuple=True)[0]
            neighbors = neighbors.tolist()
            print(neighbors)
            for neighbor in neighbors:
                if neighbor not in ratios:
                    print('No ratio for neighbor: ', neighbor)
                    continue
                ls_occ = [idx for idx, x in enumerate(labels_list) if i in x and neighbor in x]
                if len(ls_occ) == 0:
                    print('No occurence image for neighbor: ', neighbor)
                    continue
                img_name = imgs_list[ls_occ[0]]
                print(f'{i}, {neighbor}, {img_name}')
                area_i = next(x['area'] for x in annots if x['image_id'] == img_name and x['category_id'] == i)
                area_p = next(x['area'] for x in annots if x['image_id'] == img_name and x['category_id'] == neighbor)
                
                ratio_i = area_i / area_p * ratios[neighbor]
                ratios[i] = ratio_i
        if i not in ratios:
            print('No ratio for node: ', i)
            ratios[i] = 0
            
    json.dump(ratios, open('data/graph/size_ratios.json', 'w'))
        
def generate_pill_edges(pill_diagnose_path):
    pill_edges = pd.read_csv(pill_diagnose_path, names= ["pill","diagnose","weight"])
    pills = pill_edges.pill.unique()
    diags = pill_edges.diagnose.unique()
    
    pill_edges.set_index(['pill', 'diagnose'], inplace=True)
    print(pill_edges.describe())

    filtered_pill_edges = pill_edges.loc[pill_edges['weight'] > pill_edges['weight'].quantile(0.2)]
    print(filtered_pill_edges.head())
    filtered_pill_edges = filtered_pill_edges.sort_index()

    pill_pill_edges = pd.DataFrame(columns=['pill1', 'pill2', 'weight'])
    print(pill_pill_edges.head())
    for pill_a in pills:
        for pill_b in pills:
            if pill_a == pill_b:
                continue
            for diag in diags:
                if ((pill_a, diag) in filtered_pill_edges.index) and ((pill_b, diag) in filtered_pill_edges.index):
                    w1 = filtered_pill_edges.loc[(pill_a, diag)]['weight'].values[0]
                    w2 = filtered_pill_edges.loc[(pill_b, diag)]['weight'].values[0]
                    print(f'{w1} {w2}')
                    if (pill_a, pill_b) in pill_pill_edges.index:
                        pill_pill_edges.loc[(pill_a, pill_b)]['weight'] += w1 + w2
                    elif (pill_b, pill_a) in pill_pill_edges.index:
                        pill_pill_edges.loc[(pill_b, pill_a)]['weight'] += w1 + w2
                    else:
                        pill_pill_edges = pd.concat([pill_pill_edges, pd.DataFrame([[pill_a, pill_b, w1 + w2]], columns=pill_pill_edges.columns)], ignore_index=True)

    pill_pill_edges = pill_pill_edges.groupby(['pill1', 'pill2']).sum()
    print(pill_pill_edges.head())
    pill_pill_edges.to_csv('data/graph/pill_pill_graph.csv')

if __name__ == '__main__':
    # build_KG_graph('data/prescription/_merged_prescriptions_vaipe.json', name='pill_diagnose_graph')
    # merge_multilabel_meta(root='./data/pills/', train=True)
    build_size_graph(train=True)
    # print(build_size_graph_data())
    # prepare_prescription_dataset('data/prescriptions/condensed_data.json')
    # generate_pill_edges('data/graph/pill_diagnose_graph.csv')
    # condensed_result_file()
    # test()