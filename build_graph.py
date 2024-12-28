# -*- coding: utf-8 -*-

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import networkx as nx

def build_graph(type, node,dataset="junyi"):
    edge_list = []
    # 创建NetworkX图
    G = nx.Graph() if type == 'undirect' else nx.DiGraph()
    if type == 'k_from_e':
        with open(f'./data/{dataset}/graph/k_from_e_mapped.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                edge_list.append((int(line[0]), int(line[1])))
    elif type == 'e_from_k':
        with open(f'./data/{dataset}/graph/e_from_k_mapped.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                edge_list.append((int(line[0]), int(line[1])))
    elif type == 'u_from_e':
        with open(f'./data/{dataset}/graph/u_from_e_mapped.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                edge_list.append((int(line[0]), int(line[1])))
    elif type == 'e_from_u':
        with open(f'./data/{dataset}/graph/e_from_u_mapped.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                edge_list.append((int(line[0]), int(line[1])))

    # 添加节点和边到NetworkX图
    G.add_nodes_from(range(node))
    G.add_edges_from(edge_list)

    # 将NetworkX图转换为torch_geometric格式
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    
    # 如果需要无向图，则将边变为无向
    # if type == 'undirect':
    #     edge_index = to_undirected(edge_index)
    
    # 创建torch_geometric数据对象
    data = Data(edge_index=edge_index)

    return data
