import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class Fusion(nn.Module):
    def __init__(self, student_n, exer_n, knowledge_n, local_map):
        super(Fusion, self).__init__()
        self.device = torch.device('cuda:0'  if torch.cuda.is_available() else 'cpu')
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim

        # 图结构
        self.k_from_e_edge_index = local_map['k_from_e'].edge_index.to(self.device)
        self.e_from_k_edge_index = local_map['e_from_k'].edge_index.to(self.device)
        self.u_from_e_edge_index = local_map['u_from_e'].edge_index.to(self.device)
        self.e_from_u_edge_index = local_map['e_from_u'].edge_index.to(self.device)

        # GAT层
        self.k_from_e_gat = GATConv(knowledge_n, knowledge_n)
        self.e_from_k_gat = GATConv(knowledge_n, knowledge_n)
        self.u_from_e_gat = GATConv(knowledge_n, knowledge_n)
        self.e_from_u_gat = GATConv(knowledge_n, knowledge_n)

        # 注意力机制
        self.k_attn_fc3 = nn.Linear(2 * knowledge_n, 1, bias=True)
        self.e_attn_fc1 = nn.Linear(2 * knowledge_n, 1, bias=True)
        self.e_attn_fc2 = nn.Linear(2 * knowledge_n, 1, bias=True)

    def forward(self, kn_emb, exer_emb, all_stu_emb):
        # 图卷积操作
        e_k_graph = torch.cat((exer_emb, kn_emb), dim=0)
        k_from_e_graph = self.k_from_e_gat(e_k_graph, self.k_from_e_edge_index)
        e_from_k_graph = self.e_from_k_gat(e_k_graph, self.e_from_k_edge_index)

        e_u_graph = torch.cat((exer_emb, all_stu_emb), dim=0)
        u_from_e_graph = self.u_from_e_gat(e_u_graph, self.u_from_e_edge_index)
        e_from_u_graph = self.e_from_u_gat(e_u_graph, self.e_from_u_edge_index)

        # 更新知识点
        A = kn_emb
        D = k_from_e_graph[6:]   #6代表层次的个数
        concat_c_3 = torch.cat([A, D], dim=1)
        score3 = self.k_attn_fc3(concat_c_3)
        score = F.softmax( score3, dim=1)
        kn_emb = A + score[:, 0].unsqueeze(1) * D

        # 更新练习
        A = exer_emb
        B = e_from_k_graph[:6]
        C = e_from_u_graph[:6]
        concat_e_1 = torch.cat([A, B], dim=1)
        concat_e_2 = torch.cat([A, C], dim=1)
        score1 = self.e_attn_fc1(concat_e_1)
        score2 = self.e_attn_fc2(concat_e_2)
        score = F.softmax(torch.cat([score1, score2], dim=1), dim=1)
        exer_emb = exer_emb + score[:, 0].unsqueeze(1) * B + score[:, 1].unsqueeze(1) * C

        # 更新学生
        all_stu_emb = all_stu_emb + u_from_e_graph[6:]

        return kn_emb, exer_emb, all_stu_emb
