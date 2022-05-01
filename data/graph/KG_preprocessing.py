from cmath import nan
import pickle
import networkx as nx
# from torch_geometric.data import Dataset
# from torch_geometric.utils.convert import from_networkx
import json
import math
import pandas as pd
import re

def build_KG_graph(json_file, exclude_path='', name='pill_data'):
    '''
    build bipartite graph from extracted prescription json file
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
    if exclude_path != '':
        exclude_ids = pickle.load(open(exclude_path, 'rb'))
        name2idx = pickle.load(open('./data/pills/name2id.pkl', 'r'))
        exclude_names = [list(name2idx.keys())[list(name2idx.values()).index(i)] for i in exclude_ids]
    
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
                f.write(pill + ',' + diag + ',' + str(weight) + '\n')

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
                    w1 = filtered_pill_edges.loc[(pill_a, diag)]['weight']
                    w2 = filtered_pill_edges.loc[(pill_b, diag)]['weight']
                    if (pill_a, pill_b) in pill_pill_edges.index:
                        pill_pill_edges.loc[(pill_a, pill_b)]['weight'] += w1 + w2
                    elif (pill_b, pill_a) in pill_pill_edges.index:
                        pill_pill_edges.loc[(pill_b, pill_a)]['weight'] += w1 + w2
                    else:
                        row = {'pill1': pill_a, 'pill2': pill_b, 'weight': w1 + w2}
                        pill_pill_edges = pill_pill_edges.append(row, ignore_index=True)

    pill_pill_edges = pill_pill_edges.groupby(['pill1', 'pill2']).sum()
    print(pill_pill_edges.head())
    pill_pill_edges.to_csv('data/graph/pill_pill_graph.csv')

if __name__ == '__main__':
    # build_KG_graph('data/prescription/_merged_prescriptions.json', name='pill_diagnose_graph')
    # prepare_prescription_dataset('data/prescriptions/condensed_data.json')
    generate_pill_edges('data/graph/pill_diagnose_graph.csv')
    # condensed_result_file()
    # test()