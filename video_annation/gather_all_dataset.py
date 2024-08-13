import os

# 定义根目录
root_dir = r'D:\dog_video\Feb'
# 定义目标txt文件路径
target_file_path = r'D:\dog_video\all_video\combined_dataset.txt'

# 打开目标文件
with open(target_file_path, 'w', encoding='utf-8') as target_file:
    # 遍历根目录下的所有文件夹
    for subdir, dirs, files in os.walk(root_dir):
        # 检查是否是output文件夹
        if os.path.basename(subdir) == 'output':
            dataset_file_path = os.path.join(subdir, 'dataset.txt')
            # 检查dataset.txt文件是否存在
            if os.path.exists(dataset_file_path):
                # 打开dataset.txt文件并将内容写入目标文件
                with open(dataset_file_path, 'r', encoding='utf-8') as dataset_file:
                    for line in dataset_file:
                        target_file.write(line)

print(f'所有文件已成功合并到 {target_file_path}')
