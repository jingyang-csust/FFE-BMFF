# def count_common_strings(file1_path, file2_path):
#     # 读取第一个文件中的字符串
#     with open(file1_path, 'r', encoding='utf-8') as file1:
#         strings1 = set(file1.read().splitlines())
#
#     # 读取第二个文件中的字符串
#     with open(file2_path, 'r', encoding='utf-8') as file2:
#         strings2 = set(file2.read().splitlines())
#
#     # 计算相同字符串的数量
#     common_strings = strings1.intersection(strings2)
#
#     return len(common_strings)
#
#
# # 示例文件路径
# file1_path = '/root/code/lib/train/data_specs/testingsetList.txt'
# file2_path = '/root/code/lib/train/data_specs/lasher_val.txt'
#
# # 调用函数并输出结果
# common_count = count_common_strings(file1_path, file2_path)
# print(f"两个文件中有 {common_count} 个相同的字符串。")

def count_matching_lines(file1_path, file2_path):
    # 读取第二个文件中的所有字符串，并存储在一个集合中
    with open(file2_path, 'r', encoding='utf-8') as file2:
        strings2 = set(file2.read().splitlines())

    match_count = 0

    # 逐行读取第一个文件，并判断是否在第二个文件的集合中
    with open(file1_path, 'r', encoding='utf-8') as file1:
        for line in file1:
            line = line.strip()  # 去掉行末的换行符和空格
            if line in strings2:
                match_count += 1

    return match_count


# 示例文件路径
file1_path = '/root/code/lib/train/data_specs/lasher_train.txt'
file2_path = '/root/code/lib/train/data_specs/lasher_val.txt'


# 调用函数并输出结果
matching_count = count_matching_lines(file1_path, file2_path)
print(f"第一个文件中有 {matching_count} 行在第二个文件中出现。")

