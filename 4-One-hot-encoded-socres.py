
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 确保使用适用于无界面环境的后端

dataset="Read"  ###记得修改对应的数据集
# 定义处理 One-Hot 编码的函数
def process_and_generate_one_hot(file_path, output_path, score_column='Average Score'):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 计算平均分和标准差
    mean_score = df[score_column].mean()
    std_dev = df[score_column].std()

    # 基于均值和标准差来定义分数区间
    bins = [mean_score - 2 * std_dev,
            mean_score - std_dev,
            mean_score,
            mean_score + std_dev,
            mean_score + 2 * std_dev]
    bins = [0] + bins + [100]  # 保证分数区间在 0-100 范围内

    # 创建分数区间的标签
    bin_labels = ['< -2σ', '-2σ to -1σ', '-1σ to Mean', 'Mean to +1σ', '+1σ to +2σ', '> +2σ']

    # 将分数映射到区间
    df['Score Range'] = pd.cut(df[score_column], bins=bins, labels=bin_labels, include_lowest=True)

    # One-Hot 编码分数区间
    one_hot_mapping = {
        '< -2σ': 0,
        '-2σ to -1σ': 1,
        '-1σ to Mean': 2,
        'Mean to +1σ': 3,
        '+1σ to +2σ': 4,
        '> +2σ': 5
    }

    # 创建 One-Hot 编码的 DataFrame
    one_hot_encoded = pd.DataFrame(0, index=df.index, columns=one_hot_mapping.values())

    # 填充 One-Hot 编码
    for score_range, code in one_hot_mapping.items():
        one_hot_encoded.loc[df['Score Range'] == score_range, code] = 1

    # 将 One-Hot 编码添加回原始 DataFrame
    df = pd.concat([df, one_hot_encoded], axis=1)

    # 重命名 One-Hot 编码列
    df.columns = list(df.columns[:-len(one_hot_mapping)]) + list(one_hot_mapping.keys())

    # 添加十进制编码列（将 One-Hot 编码列转换为十进制数）
    df['Score Range Decimal'] = df[bin_labels].dot([one_hot_mapping[label] for label in bin_labels])

    # 保存包含 One-Hot 编码和十进制编码的 DataFrame 到 CSV 文件
    df.to_csv(output_path, index=False)


# 处理 average_scores_with_stu_id.csv
process_and_generate_one_hot(f'./data/{dataset}/average_scores_with_stu_id.csv', f'./data/{dataset}/stu_one_hot_encoded_scores.csv')

# 处理 average_scores_per_question.csv
process_and_generate_one_hot(f'./data/{dataset}/average_scores_per_question.csv', f'./data/{dataset}/question_one_hot_encoded.csv', score_column='Average Score')

# 处理 average_knowledge_mastery.csv
process_and_generate_one_hot(f'./data/{dataset}/average_knowledge_mastery.csv', f'./data/{dataset}/knowledge_one_hot_encoded.csv', score_column='Average Mastery')


