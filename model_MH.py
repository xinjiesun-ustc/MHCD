import torch
import torch.nn as nn
import numpy as np
import random
from fusion import Fusion

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(1)

# 创建激活函数层
relu = torch.nn.ReLU()
# tanh_layer = nn.Tanh()
tanh_layer = nn.Sigmoid()

class Net(nn.Module):
    '''
    NeuralCDM
    '''
    def __init__(self, student_n, exer_n, knowledge_n, local_map):
        super(Net, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, self.knowledge_dim  # 可调整

        # 新增
        self.theta_bias = nn.Embedding(self.emb_num, 1, padding_idx=0)
        self.theta_bias_2 = nn.Embedding(self.emb_num, 1, padding_idx=0)

        # 可训练的 beta 和 alpha 参数
        self.beta = nn.Parameter(torch.tensor(1.0))  # 初始值为 1.0
        self.alpha = nn.Parameter(torch.tensor(1.0))  # 初始值为 1.0

        # 图结构映射
        self.k_from_e = local_map['k_from_e'].to(self.device)
        self.e_from_k = local_map['e_from_k'].to(self.device)
        self.u_from_e = local_map['u_from_e'].to(self.device)
        self.e_from_u = local_map['e_from_u'].to(self.device)



        # 融合层
        self.FusionLayer1 = Fusion(student_n, exer_n, knowledge_n, local_map)
        self.FusionLayer2 = Fusion(student_n, exer_n, knowledge_n, local_map)

        # 层次嵌入
        self.student_hierarchy_emb = nn.Embedding(6, self.stu_dim)
        self.knowledge_hierarchy_emb = nn.Embedding(6, self.knowledge_dim)
        self.exercise_hierarchy_emb = nn.Embedding(6, self.knowledge_dim)


        # 学生、知识点、练习嵌入
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)

        # 预测网络
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.6)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.6)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, input_knowledge_onehot_embs, stu_hierarchy_id, exer_hierarchy_id):
        # 提前将权重转移到设备并缓存
        all_stu_hierarchy_emb = self.student_hierarchy_emb.weight.to(self.device)
        exer_hierarchy_emb = self.exercise_hierarchy_emb.weight.to(self.device)
        kn_hierarchy_emb = self.knowledge_hierarchy_emb.weight.to(self.device)

        # 融合层
        kn_hierarchy_emb1, exer_hierarchy_emb1, all_stu_hierarchy_emb1 = self.FusionLayer1(kn_hierarchy_emb, exer_hierarchy_emb, all_stu_hierarchy_emb)
        kn_hierarchy_emb2, exer_hierarchy_emb2, all_stu_hierarchy_emb2 = self.FusionLayer2(kn_hierarchy_emb1, exer_hierarchy_emb1, all_stu_hierarchy_emb1)

        # 获取 batch 数据嵌入
        batch_stu_hierarchy_emb = all_stu_hierarchy_emb2[stu_hierarchy_id]
        batch_exer_hierarchy_emb = exer_hierarchy_emb2[exer_hierarchy_id]

        # 学生、知识点、练习嵌入
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10

        # 影响权重系数
        con_belif = torch.sigmoid(self.theta_bias(stu_id)).view(-1, 1)
        ablity_belif = torch.sigmoid(self.theta_bias_2(stu_id)).view(-1, 1)

        # 输入计算
        input_x1 = e_discrimination * (con_belif * stu_emb + (1 - con_belif) * batch_stu_hierarchy_emb - ablity_belif * k_difficulty - (1 - ablity_belif) * batch_exer_hierarchy_emb) * input_knowledge_onehot_embs

        input_x2 = self.drop_1(tanh_layer(self.prednet_full1(input_x1)))
        input_x3 = self.drop_2(tanh_layer(self.prednet_full2(input_x2)))
        output = torch.sigmoid(self.prednet_full3(input_x3))
        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
