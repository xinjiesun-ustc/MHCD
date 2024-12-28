import pandas as pd
import os
dataset="Read"  ###记得修改对应的数据集
def load_mappings(question_file, knowledge_file):
    """
    加载题目和知识点的编号到层次映射。
    Args:
        question_file (str): 包含题目信息的 CSV 文件路径。
        knowledge_file (str): 包含知识点信息的 CSV 文件路径。
    Returns:
        tuple: 题目和知识点的编号到层次的映射字典。
    """
    # 读取文件并创建编号到层次的映射
    question_df = pd.read_csv(question_file, header=0, encoding='ISO-8859-1')
    knowledge_df = pd.read_csv(knowledge_file, header=0, encoding='ISO-8859-1')
    question_mapping = question_df.set_index(question_df.columns[0])[question_df.columns[-1]].to_dict()
    knowledge_mapping = knowledge_df.set_index(knowledge_df.columns[0])[knowledge_df.columns[-1]].to_dict()
    return question_mapping, knowledge_mapping

def replace_ids_with_levels(input_file, output_file, src_mapping, dst_mapping, offset=6, stu_mapping=None):
    """
    替换文件中编号为映射值，并保存到新的文件中，去除重复行。
    Args:
        input_file (str): 输入文件路径，文件内容为源编号和目标编号。
        output_file (str): 输出文件路径。
        src_mapping (dict): 源编号到映射值的字典。
        dst_mapping (dict): 目标编号到映射值的字典。
        offset (int): 如果映射值来自知识点映射，添加的偏移量。
        knowledge_mapping (dict): 知识点的映射值，用于判断是否需要加偏移量。
    """
    if not os.path.exists(input_file):
        print(f"文件 {input_file} 不存在，跳过处理。")
        return

    # 读取输入文件内容
    with open(input_file, 'r') as f:
        lines = f.readlines()

    print(f"处理文件: {input_file}, 总行数: {len(lines)}")

    # 用集合跟踪唯一的映射对
    seen_pairs = set()

    # 替换并写入到输出文件
    with open(output_file, 'w') as f:
        for line in lines:
            try:
                # 强制将 src 和 dst 转换为整数
                src, dst = map(int, line.strip().split('\t'))
                new_src = src_mapping.get(src, src)
                new_dst = dst_mapping.get(dst, dst)

                # 如果 src 或 dst 来自 knowledge_mapping，添加偏移量
                if stu_mapping and src in stu_mapping:
                    new_src += offset
                if stu_mapping and dst in stu_mapping:
                    new_dst += offset

                # 如果该对已经存在，跳过
                if (new_src, new_dst) in seen_pairs:
                    continue

                # 记录到集合并写入文件
                seen_pairs.add((new_src, new_dst))
                f.write(f"{new_src}\t{new_dst}\n")
            except ValueError as e:
                print(f"跳过无效行: {line.strip()} - {e}")

    print(f"文件已处理并保存到: {output_file}")

if __name__ == '__main__':
    # 输入文件路径
    question_file = f'./data/{dataset}/question_one_hot_encoded.csv'
    stu_file = f'./data/{dataset}/stu_one_hot_encoded_scores.csv'

    # 加载映射关系
    question_mapping, stu_mapping = load_mappings(question_file, stu_file)

    # 处理并替换文件内容
    replace_ids_with_levels(
        f'./data/{dataset}/graph/u_from_e.txt',
        f'./data/{dataset}/graph/u_from_e_mapped.txt',
        question_mapping,
        stu_mapping,
        offset=6,
        stu_mapping=stu_mapping  # 传入知识点映射
    )
    replace_ids_with_levels(
        f'./data/{dataset}/graph/e_from_u.txt',
        f'./data/{dataset}/graph/e_from_u_mapped.txt',
        stu_mapping,
        question_mapping,
        offset=6,
        stu_mapping=stu_mapping  # 传入知识点映射
    )

    print("所有映射操作完成。")
