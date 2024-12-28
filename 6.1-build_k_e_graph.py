import json
import torch
import os
from torch_geometric.data import Data

def build_local_map(dataset):
    data_file = f'./data/{dataset}/Readlog_data.json'
    with open(f'./data/{dataset}/config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    temp_set = set()  # 使用集合来避免重复
    # ##sun注释：：：：N:M的多重关系，一个题目包含多个知识点，一个知识点也属于多个题目
    edges_k_from_e = []  # e(src) -> k(dst) 的边
    edges_e_from_k = []  # k(src) -> e(dst) 的边

    with open(data_file, encoding='utf8') as i_f:
        data = json.load(i_f)
    
    for line in data:
        for log in line['logs']:
            exer_id = log['exer_id']  #确保从1开始索引
            for k in log['knowledge_code']:
                knowledge_id = k + exer_n  # 知识点的ID需要加上练习数量的偏移量
                edge_k_from_e = (exer_id, knowledge_id)
                edge_e_from_k = (knowledge_id, exer_id)

                # 防止重复边
                if edge_k_from_e not in temp_set:
                    edges_k_from_e.append(edge_k_from_e)
                    edges_e_from_k.append(edge_e_from_k)
                    temp_set.add(edge_k_from_e)
                    temp_set.add(edge_e_from_k)

    # 将边转化为torch tensor
    edge_index_k_from_e = torch.tensor(edges_k_from_e, dtype=torch.long).t().contiguous()
    edge_index_e_from_k = torch.tensor(edges_e_from_k, dtype=torch.long).t().contiguous()

    # 构建torch_geometric的数据对象
    data_k_from_e = Data(edge_index=edge_index_k_from_e)
    data_e_from_k = Data(edge_index=edge_index_e_from_k)

    # 保存为.txt文件
    if not os.path.exists(f"./data/{dataset}/graph/"):
        os.makedirs(f"./data/{dataset}/graph/")
    with open(f'./data/{dataset}/graph/k_from_e.txt', 'w') as f:
        for edge in edges_k_from_e:
            f.write(f"{edge[0]}\t{edge[1]}\n")

    with open(f'./data/{dataset}/graph/e_from_k.txt', 'w') as f:
        for edge in edges_e_from_k:
            f.write(f"{edge[0]}\t{edge[1]}\n")

    return data_k_from_e, data_e_from_k  # 返回图数据对象供后续使用

if __name__ == '__main__':
    
    # build_local_map(dataset='Science')
    build_local_map(dataset='Read')
    # build_local_map(dataset="Math")
