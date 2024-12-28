


import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')  # 或者使用 'TkAgg'
import matplotlib.pyplot as plt


offect =103   ##定义偏移量=题目的个数  其他所有数据集记得修改为对应的数据集的题目个数
dataset="Read"  ###记得修改对应的数据集
# 读取所有题目的数据
file_path_questions = "./data/pisa-read-2015-transform_to_number.csv"
df = pd.read_csv(file_path_questions, low_memory=False)

# 读取需要的题目 ID
file_path_ids = "./data/pisa2015-read-kcs.csv"
df_ids = pd.read_csv(file_path_ids, low_memory=False)

# 提取需要的题目 ID 列
required_ids = df_ids['ID'].unique()  # 假设 ID 列名为 'ID'

# 排除指定的字段
excluded_columns = ['STU_ID']
question_ids = [col for col in df.columns if col not in excluded_columns]

# 筛选出所需题目
filtered_question_ids = [qid for qid in question_ids if qid in required_ids]

# 确保包括 STU_ID 列和筛选后的题目列
df_new = df.loc[:, ['STU_ID'] + filtered_question_ids]

# 清理数据，删除没有有效回答的列
df_question = df_new.loc[:, filtered_question_ids]
column_counts = df_question.count()
columns_with_zero_count = column_counts[column_counts == 0].index
df_question_cleaned = df_question.drop(columns=columns_with_zero_count)

# 计算每个学生的答题数量
df_new['Question Count'] = df_question_cleaned.notnull().sum(axis=1)

# 筛选答题数量大于或等于 30 条有效记录的学生
df_student_filtered = df_new[df_new['Question Count'] >= 30].copy()

# 计算筛选后学生的有效总答题数量
total_filtered_question_count = df_student_filtered['Question Count'].sum()

# 输出结果
print("删除学习序列较短的学生后，还有{}人".format(len(df_student_filtered)))
print("删除学习序列较短的学生后，还有{}个有效答题数".format(total_filtered_question_count))

# 计算筛选后 df 中值为 1 和 0 的题目数量
count_of_zeros_in_filtered = (df_student_filtered[filtered_question_ids] == 0).sum()
count_of_ones_in_filtered = (df_student_filtered[filtered_question_ids] == 1).sum()

# 输出结果
print(f"筛选后为 0 的题目数量: {count_of_zeros_in_filtered.sum()}")
print(f"筛选后为 1 的题目数量: {count_of_ones_in_filtered.sum()}")

print("-----------------------------------------------------------------------------------------------")

print("开始处理筛选后的数据，保存为 JSON 格式……")

# 加载 exer_id_mapping.json
with open('./data/exer_id_mapping.json', 'r') as file:
    exer_id_mapping = json.load(file)

# 加载 Q_matrix
Q_matrix = np.load('./data/Q_matrix.npy')

# 重新映射 STU_ID
stu_id_mapping = {stu_id: i + 1 for i, stu_id in enumerate(df_student_filtered['STU_ID'].unique())}
df_student_filtered.loc[:, 'STU_ID'] = df_student_filtered['STU_ID'].map(stu_id_mapping)

# 计算每个学生的平均得分
df_student_filtered.loc[:, 'Average Score'] = df_student_filtered[filtered_question_ids].mean(axis=1)

# 获取 STU_ID 和 Average Score
average_scores_with_stu_id = df_student_filtered.loc[:, ['STU_ID', 'Average Score']].copy()

# # 将 STU_ID 列减去 1
average_scores_with_stu_id['STU_ID'] = average_scores_with_stu_id['STU_ID'] + offect

# 将 STU_ID 和平均得分保存到磁盘
average_scores_with_stu_id.to_csv(f'./data/{dataset}/average_scores_with_stu_id.csv', index=False)

# 新增功能：统计每个题目的平均得分
print("统计每个题目的平均得分……")
average_scores_per_question = df_student_filtered[filtered_question_ids].mean(axis=0)

# # 将题目 ID 和对应的平均得分保存到 CSV 文件
# average_scores_per_question_df = average_scores_per_question.reset_index()
# average_scores_per_question_df.columns = ['Question ID', 'Average Score']
# average_scores_per_question_df.to_csv('./data/average_scores_per_question.csv', index=False)

# 加载 exer_id_mapping.json
with open('./data/exer_id_mapping.json', 'r') as f:
    exer_id_mapping = json.load(f)

# 将 Question ID 映射为对应的值
average_scores_per_question_df = average_scores_per_question.reset_index()
average_scores_per_question_df.columns = ['Question ID', 'Average Score']

# 替换 Question ID
average_scores_per_question_df['Question ID'] = average_scores_per_question_df['Question ID'].map(exer_id_mapping)

# 保存到 CSV 文件
output_file = f'./data/{dataset}/average_scores_per_question.csv'
average_scores_per_question_df.to_csv(output_file, index=False)

print(f"已将平均得分保存到文件: {output_file}")


# 输出结果
print(f"每个题目的平均得分已保存到 ./data/{dataset}/average_scores_per_question.csv")


# 处理每一行数据
logs = []
for index, row in df_student_filtered.iterrows():
    user_id = row['STU_ID']
    user_logs = []
    valid_answer_count = 0  # 计数有效答案

    for column in filtered_question_ids:  # 只处理题目相关列
        exer_id = exer_id_mapping.get(column)  # 获取题目编号
        score = row[column]  # 已经是数字

        if exer_id is not None and not np.isnan(score):
            # 获取知识点
            knowledge_points = np.where(Q_matrix[exer_id - 1] > 0)[0] + 1  # 获取知识点编号
            user_logs.append({
                "exer_id": exer_id,
                "score": score,
                "knowledge_code": knowledge_points.tolist()  # 转换为列表
            })
            # if score in [0, 1, 2]:  # 计算有效的答题个数
            valid_answer_count += 1

    logs.append({
        "user_id": int(user_id),
        "log_num": valid_answer_count,  # 有效答题个数
        "logs": user_logs
    })

# 保存到 JSON 文件
with open(f'data/Read/{dataset}log_data.json', 'w') as outfile:
    json.dump(logs, outfile, indent=4)

print("处理完成，数据已保存为 JSON 格式。")