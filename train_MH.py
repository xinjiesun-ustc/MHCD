import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys

import pandas as pd
import torch

from sklearn.metrics import roc_auc_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model_MH import Net
import matplotlib
from utils import construct_local_map
from tqdm import tqdm

matplotlib.use('Agg')  # 或者使用 'TkAgg'

# can be changed according to config.txt
exer_n = 81
knowledge_n = 11
student_n = 116805
device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
epoch_n = 5

dataset = "Math"

# 预加载所有 CSV 文件
print("Loading CSV files...")
df_stu = pd.read_csv(f'./data/{dataset}/stu_hierarchy_map_基于onehot手动构建.csv')
df_exer = pd.read_csv(f'./data/{dataset}/question_Hierarchy_map_基于onehot手动构建.csv')



# 构造加速查找的字典
def construct_hierarchy_dict(df):
    return dict(zip(df[df.columns[0]], df[df.columns[1]]))

hierarchy_stu_dict = construct_hierarchy_dict(df_stu)
hierarchy_exer_dict = construct_hierarchy_dict(df_exer)


# 使用字典进行快速查找
def get_hierarchy_by_ids_fast(hierarchy_dict, stu_ids):
    return [hierarchy_dict.get(stu_id.item(), "IDs not found") for stu_id in stu_ids]


def train(Local_map):
    data_loader = TrainDataLoader()
    net = Net(student_n, exer_n, knowledge_n, Local_map)
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    print('Training model...')

    loss_function = nn.NLLLoss()

    for epoch in range(epoch_n):
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0

        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_onehot_embs, labels = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_onehot_embs, labels = ( input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_onehot_embs.to(device), labels.to(device),)
            optimizer.zero_grad()

            # 获取层次 ID
            stu_hierarchy_id = get_hierarchy_by_ids_fast(hierarchy_stu_dict, input_stu_ids)
            exer_hierarchy_id = get_hierarchy_by_ids_fast(hierarchy_exer_dict, input_exer_ids)

            output_1 = net.forward(input_stu_ids, input_exer_ids, input_knowledge_onehot_embs, stu_hierarchy_id, exer_hierarchy_id )
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)

            loss = loss_function(torch.log(output), labels)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0

        rmse, auc = validate(net, epoch, Local_map)
        save_snapshot(net, 'model/Read_model_MH_epoch' + str(epoch + 1))


def validate(model, epoch, Local_map):
    data_loader = ValTestDataLoader('test')
    net = Net(student_n, exer_n, knowledge_n, Local_map)
    print('Validating model...')

    data_loader.reset()
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    pred_all, label_all = [], []

    # 手动计算总批次数量
    batch_size = data_loader.batch_size if hasattr(data_loader, 'batch_size') else 1
    total_samples = len(data_loader.data) if hasattr(data_loader, 'data') else 1
    total_batches = (total_samples + batch_size - 1) // batch_size

    with tqdm(total=total_batches, desc=f"Validating Epoch {epoch+1}") as pbar:
        while not data_loader.is_end():
            input_stu_ids, input_exer_ids, input_knowledge_onehot_embs, labels = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_onehot_embs, labels = (
                input_stu_ids.to(device),
                input_exer_ids.to(device),
                input_knowledge_onehot_embs.to(device),
                labels.to(device),
            )

            stu_hierarchy_id = get_hierarchy_by_ids_fast(hierarchy_stu_dict, input_stu_ids)
            exer_hierarchy_id = get_hierarchy_by_ids_fast(hierarchy_exer_dict, input_exer_ids)

            output = net.forward(
                input_stu_ids, input_exer_ids, input_knowledge_onehot_embs, stu_hierarchy_id, exer_hierarchy_id
            )
            output = output.view(-1)

            for i in range(len(labels)):
                if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                    correct_count += 1
            exer_count += len(labels)
            pred_all += output.to(torch.device('cpu')).tolist()
            label_all += labels.to(torch.device('cpu')).tolist()

            # 更新进度条
            pbar.update(1)

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    accuracy = correct_count / exer_count
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch + 1, accuracy, rmse, auc))
    with open('result/Read_model_MH_test.txt', 'a', encoding='utf8') as f:
        f.write('1, epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch + 1, accuracy, rmse, auc))

    return rmse, auc


def save_snapshot(model, filename):
    with open(filename, 'wb') as f:
        torch.save(model.state_dict(), f)

if __name__ == '__main__':
    if (len(sys.argv) != 3) or ((sys.argv[1] != 'cpu') and ('cuda:' not in sys.argv[1])) or (not sys.argv[2].isdigit()):
        print('command:\n\tpython train.py {device} {epoch}\nexample:\n\tpython train.py cuda:0 70')
        exit(1)
    else:
        device = torch.device(sys.argv[1])
        epoch_n = int(sys.argv[2])

    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    Local_map = construct_local_map(dataset)
    train(Local_map)