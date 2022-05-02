import pickle
import torch
import torch_geometric
from torch_geometric.data import Data
import json
import pandas as pd
import networkx as nx
import config as CFG
import numpy as np
import dgl
from torch_geometric.utils import to_dense_adj
from scipy.sparse import rand, coo_matrix

def convert_2_homogeneous_graph():
    df_pill = pd.read_csv('data/prescriptions/pill_data.dat', names= ["pill","prescription","tfidf"], sep = "\\")
    df_pill['tfidf'] = df_pill['tfidf'].astype(float)
    pill_list_unique = df_pill["pill"].unique()
    prescription_list_unique = df_pill['prescription'].unique()
    

def build_data():
    mapped_pill_idx = json.load(open('data/converted_graph/mapdict.json', 'r'))
    baseline_weights = torch.load(CFG.backbone_path)
    classifier_w = baseline_weights['classifier.weight']
    # print(classifier_w.shape)
    edge_index = []
    edge_weight = []
    
    mapped_name = pd.read_csv('data/converted_graph/mapped_name.csv', header=0)

    pill_edge = pd.read_csv('data/converted_graph/pill_edge.dat', names= ["pillA","pillB","tfidf"], sep = "@")
    pill_edge['pillA'] = pill_edge['pillA'].map(lambda a : mapped_name[mapped_name['pill'] == a].values[0][-1])
    pill_edge['pillB'] = pill_edge['pillB'].map(lambda a : mapped_name[mapped_name['pill'] == a].values[0][-1])
    # print(pill_edge.head(5))
    pill_edge.dropna(inplace=True)

    for x, y, w in pill_edge.values:
        # print(x)
        # print(y)
        # print(w)
        assert(w > 0)
        edge_index.append([mapped_pill_idx[x], mapped_pill_idx[y]])
        edge_weight.append(w)
        edge_index.append([mapped_pill_idx[y], mapped_pill_idx[x]])
        edge_weight.append(w)
    
    data = Data(x=torch.eye(CFG.n_class, dtype=torch.float32), edge_index=torch.tensor(edge_index).t().contiguous(), edge_attr=torch.tensor(edge_weight).unsqueeze(1))
    # print(data)
    return data, to_dense_adj(data.edge_index, edge_attr=data.edge_attr).squeeze()

def build_data_dgl():
    '''
    build graph dataset in form of networkx graph
    '''
    mapped_pill_idx = json.load(open('data/converted_graph/mapdict.json', 'r'))
    baseline_weights = torch.load(CFG.backbone_path)
    # classifier_w = baseline_weights['classifier.weight'].to('cpu')
    classifier_w = torch.eye(CFG.n_class, dtype=torch.float32)
    print(classifier_w.shape)
    edge_src = []
    edge_dst = []
    edge_weight = []

    pill_edge = pd.read_csv('data/converted_graph/mapped_pill_edges.csv', header=0)
    # print(pill_edge.head(5))
    for x, y, w in pill_edge.values:
        assert(w > 0)
        edge_src.append(mapped_pill_idx[x])
        edge_dst.append(mapped_pill_idx[y])
        edge_weight.append(w)

    graph = dgl.graph((edge_src, edge_dst), num_nodes=classifier_w.shape[0])
    graph.ndata['feat'] = classifier_w
    graph.edata['weight'] = torch.tensor(edge_weight, dtype=torch.float32)

    graph_data = (graph.edata['weight'].squeeze(-1), graph.edges())
    shape = (graph.num_nodes(), graph.num_nodes())
    adj = coo_matrix(graph_data, shape).toarray()
    
    graph = dgl.add_self_loop(graph)
    return graph, dgl.to_networkx(graph, node_attrs=['feat'], edge_attrs=['weight']), adj

def build_adj_matrix():
    mapped_pill_idx = json.load(open('data/converted_graph/mapdict.json', 'r'))
    mapped_name = pd.read_csv('data/converted_graph/mapped_name.csv', header=0)
    adj_matrix = np.zeros((CFG.n_class, CFG.n_class))

    pill_edge = pd.read_csv('data/converted_graph/pill_edge.dat', names= ["pillA","pillB","tfidf"], sep = "@")
    pill_edge['pillA'] = pill_edge['pillA'].map(lambda a : mapped_name[mapped_name['pill'] == a].values[0][-1])
    pill_edge['pillB'] = pill_edge['pillB'].map(lambda a : mapped_name[mapped_name['pill'] == a].values[0][-1])
    # print(pill_edge.head(5))
    pill_edge.dropna(inplace=True)

    for x, y, w in pill_edge.values:
        adj_matrix[mapped_pill_idx[x], mapped_pill_idx[y]] = w
    
    import pickle
    with open('data/converted_graph/pill_adj_matrix.pkl', 'wb') as handle:
        pickle.dump(adj_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return adj_matrix

def visualize_graph(graph=None):
    '''
    visualize networkx graph
    '''
    if graph is None:
        mapped_pill_idx = pickle.load(open('data/pills/name2id.pkl', 'rb'))
        xs = mapped_pill_idx.keys()
        edge_index = []
        edge_weight = []

        pill_edge = pd.read_csv('data/graph/pill_pill_graph.csv', header=0)
        # print(pill_edge.head(5))
        pill_edge.dropna(inplace=True)

        for x, y, w in pill_edge.values:
            # print(x)
            # print(y)
            # print(w)
            if x in xs and y in xs:
                edge_index.append([mapped_pill_idx[x], mapped_pill_idx[y], w])
                edge_weight.append(w)
                edge_index.append([mapped_pill_idx[y], mapped_pill_idx[x], w])
                edge_weight.append(w)
            # if mapped_pill_idx[x] == 11 or mapped_pill_idx[y] == 11:
            #     print(f'{mapped_pill_idx[x]} {mapped_pill_idx[y]} {w}')
        
        edge_weight = np.array(edge_weight)
        
        # draw edge weight distribution
        # fig = plt.figure(figsize =(10, 7))
        # # Creating axes instance
        # ax = fig.add_axes([0, 0, 1, 1])
        # ax.ylabel('Edge weight')
        # # Creating plot
        # bp = ax.boxplot(edge_weight)
        # fig.savefig('logs/imgs/boxplot_edges.png') 
        
        # fig = plt.figure(figsize =(5, 1.5))
        # # Creating axes instance
        # import seaborn as sns
        # sns.boxplot(data=edge_weight, orient='h', palette='pastel')
        # # Creating plot
        # # bp = ax.boxplot(edge_weight, vert=False)
        # plt.xlabel('Edge weight')
        # plt.yticks([])
        # plt.margins(y=0.5)
        # plt.tight_layout(pad=0.2)
        # fig.savefig(CFG.log_dir_data + 'boxplot_edges.png', dpi=150) 
        # plt.clf()
        
        print(f'weight max: {np.max(edge_weight)}, min: {np.min(edge_weight)}, mean: {np.mean(edge_weight)}, std: {np.std(edge_weight)}')
        # data = Data(x=xs, edge_index=torch.tensor(edge_index).t().contiguous(), edge_attr=torch.tensor(edge_weight))
        G = nx.Graph()
        G.add_weighted_edges_from(edge_index)
    else:
        G = graph

    # print(G.edges.data('weight'))
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > np.quantile(edge_weight, 0.7)]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= np.quantile(edge_weight, 0.7)]

    pos = nx.spring_layout(G, seed=911, k=1.5)  # positions for all nodes - seed for reproducibility
    # pos = nx.graphviz_layout(G)
    # ill_pred = [1, 11, 14, 15, 20, 27, 31, 32, 36, 41, 42, 52, 56, 66, 75]
    # values = [0.75 if node not in ill_pred else 1.0 for node in G.nodes()]
    # nodes
    # nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('viridis'), node_color=values, node_size=200)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('viridis'), node_size=200)
    # edges
    nx.draw_networkx_edges(
        G, pos, edgelist=esmall, width=0.1, alpha=0.5, edge_color="b", style="dashed"
    )
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=1)

    # labels
    nx.draw_networkx_labels(G, pos, font_color='w', font_size=7, font_family="sans-serif")

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(CFG.log_dir_data + 'graph.png', dpi=100)

def get_hidden_states(encoded, token_ids_word, model, layers):
    """Push input IDs through model. Stack and sum `layers` (last four by default).
    Select only those subword token outputs that belong to our word of interest
    and average them."""
    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
    # Only select the tokens that constitute the requested word
    word_tokens_output = output[token_ids_word]

    return word_tokens_output.mean(dim=0)
 

def get_word_vector(sent, idx, tokenizer, model, layers):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
        that make up the word of interest, and then `get_hidden_states`."""
    encoded = tokenizer.encode_plus(sent, return_tensors="pt")
    # get all token idxs that belong to the word of interest
    token_ids_word = np.where(np.array(encoded.word_ids()) == idx)

    return get_hidden_states(encoded, token_ids_word, model, layers)

def build_label_embedding():
    from transformers import AutoTokenizer, AutoModel
    mapped_pill_idx = json.load(open('data/converted_graph/mapdict.json', 'r'))
    names = mapped_pill_idx.keys()
    
    layers = [-4, -3, -2, -1]
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biosyn-sapbert-bc5cdr-chemical') # biobert for chemical - mostly resemble Drugname
    model = AutoModel.from_pretrained('dmis-lab/biosyn-sapbert-bc5cdr-chemical', output_hidden_states=True)

    word_ebd = []
    for sent in names:
        word_ebd.append(get_word_vector(sent, 0, tokenizer, model, layers))

    word_ebd = torch.stack(word_ebd).numpy()
    print(word_ebd.shape)
    np.save('data/converted_graph/pill_word_ebd.npy', word_ebd)
    
def generate_exclude_list(n_class=CFG.n_classes):
    import random
    import pickle
    
    ls_num = set(range(n_class))
    # preserve 25 class for test
    test_list = set(random.sample(ls_num, 25)) 
    
    n_class = n_class - len(test_list)
    n_25 = n_class // 4
    ls_num = ls_num - test_list
    print(test_list)
    with open('./data/converted_graph/graph_exp2/test_list.pkl', 'wb') as f:
        pickle.dump(test_list, f)
        
    q_25 = set(random.sample(ls_num, n_25))
    print(q_25)
    with open('./data/converted_graph/graph_exp2/exclude_25.pkl', 'wb') as f:
        pickle.dump(q_25, f)
    
    n_50 = n_class // 2 - n_25
    q_50 = set(random.sample(ls_num - q_25, n_50)).union(q_25)
    print(q_50)
    with open('./data/converted_graph/graph_exp2/exclude_50.pkl', 'wb') as f:
        pickle.dump(q_50, f)
        
    n_75 = n_class * 3 // 4 - n_25 - n_50
    q_75 = set(random.sample(ls_num - q_50, n_75)).union(q_50)
    print(q_75)
    with open('./data/converted_graph/graph_exp2/exclude_75.pkl', 'wb') as f:
        pickle.dump(q_75, f)

def generate_graph_tensor():
    name2idx = pickle.load(open('./data/pills/name2id.pkl', 'rb'))
    graph = pd.read_csv('data/graph/vectors_u.csv', sep='\\', names=['pill', 'ebd'])

    graph_tensor = [[] for _ in range(CFG.n_classes)]
    for pill, ebd in graph.values:
        if pill in name2idx:
            idx = name2idx[pill]
            ebd = ebd.strip().split(' ')
            graph_tensor[idx] = [float(x) for x in ebd]
    
    graph_tensor = torch.tensor(graph_tensor, dtype=torch.float32)
    graph_tensor = torch.cat([graph_tensor, torch.zeros((1,128), dtype=torch.float32)], dim=0)
    print(graph_tensor.shape)
    torch.save(graph_tensor, './data/graph/graph_ebd.pt')
    
if __name__ == '__main__':
    # build_data()
    import matplotlib.pyplot as plt
    # g, g_nx, a_mtx = build_data_dgl()
    # print(g_nx.edges)
    # g_nx = g_nx.to_undirected()
    # visualize_graph()
    generate_graph_tensor()
    # generate_exclude_list()
    # build_label_embedding()
    # build_adj_matrix()
    # build_data()
    # import seaborn as sns

    # ax = sns.heatmap(a_mtx / (1e-6 + a_mtx.sum(axis=1, keepdims=True)), cmap='viridis', vmin=0, vmax=1)
    # plt.savefig(CFG.log_dir_data + 'graph_adj_heatmap.png', dpi=100)