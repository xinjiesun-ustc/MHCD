import json
import torch
import os
from torch_geometric.data import Data

def build_local_map(dataset):
    data_file = f'./data/{dataset}/train_set.json'
    with open(f'./data/{dataset}/config.txt') as i_f:
        i_f.readline()  # 跳过第一行
        student_n, exer_n, knowledge_n = list(map(int, i_f.readline().split(',')))


####sun注释：每一个题目别多个学生拥有，每一个学生拥有多个题目
    edges_u_from_e = []  # e(src) -> u(dst) 的边
    edges_e_from_u = []  # u(src) -> e(dst) 的边

    # 读取训练集数据文件
    with open(data_file, encoding='utf8') as i_f:
        data = json.load(i_f)

    print(len(data))  # 输出数据集大小
    for line in data:
        exer_id = line['exer_id']  # 练习ID，从1开始编号
        user_id = line['user_id'] # 用户ID，从1开始编号
        for k in line['knowledge_code']:
            # e -> u: 从练习到用户
            edges_u_from_e.append((exer_id, user_id + exer_n))  ##用户编码多加一个题目编号 保证源和目的地的数字不会重复
            # u -> e: 从用户到练习
            edges_e_from_u.append((user_id + exer_n, exer_id))

    # 将边转换为torch tensor，并转换为适合torch_geometric的格式
    edge_index_u_from_e = torch.tensor(edges_u_from_e, dtype=torch.long).t().contiguous()
    edge_index_e_from_u = torch.tensor(edges_e_from_u, dtype=torch.long).t().contiguous()

    # 构建torch_geometric的数据对象
    data_u_from_e = Data(edge_index=edge_index_u_from_e)
    data_e_from_u = Data(edge_index=edge_index_e_from_u)

    # 将边保存到文件
    if not os.path.exists(f"./data/{dataset}/graph/"):
        os.makedirs(f"./data/{dataset}/graph/")
    with open(f'./data/{dataset}/graph/u_from_e.txt', 'w') as f_u_from_e:
        for edge in edges_u_from_e:
            f_u_from_e.write(f"{edge[0]}\t{edge[1]}\n")

    with open(f'./data/{dataset}/graph/e_from_u.txt', 'w') as f_e_from_u:
        for edge in edges_e_from_u:
            f_e_from_u.write(f"{edge[0]}\t{edge[1]}\n")

    return data_u_from_e, data_e_from_u  # 返回图数据对象供后续使用

if __name__ == '__main__':
    # build_local_map(dataset='Science')
    build_local_map(dataset='Read')
    # build_local_map(dataset="Math")