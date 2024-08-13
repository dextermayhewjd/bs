import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd

# 设置主文件夹路径
main_dir = r'D:\dog_video\Feb\2_2_kt\all_good'
mp4_dir = os.path.join(main_dir, 'all')
dataset_file = os.path.join(main_dir, 'dataset.txt')

# 读取dataset.txt文件
data = []
with open(dataset_file, 'r') as f:
    for line in f:
        parts = line.strip().rsplit(' ', 1)
        mp4_file, labels = parts[0], parts[1]
        data.append((mp4_file, labels))

# 创建数据框架
df = pd.DataFrame(data, columns=['filename', 'labels'])
print(f"总数据大小: {len(df)}")  # 调试信息

# 分开处理标签并计算每个标签的出现次数
all_labels = df['labels'].str.split(' ').explode()
label_counts = all_labels.value_counts()

# 筛选出出现次数大于8的标签
valid_labels = label_counts[label_counts > 8].index
print(f"有效标签: {valid_labels}")  # 调试信息

# 过滤数据框，只保留包含有效标签的行
def has_valid_label(labels, valid_labels):
    return any(label in valid_labels for label in labels.split('_'))

df = df[df['labels'].apply(has_valid_label, valid_labels=valid_labels)]
print(f"过滤后数据大小: {len(df)}")  # 调试信息

# 按照7:2:1的比例分割数据集
train_files, temp_files = train_test_split(df, test_size=0.3, stratify=df['labels'], random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=2/3, stratify=temp_files['labels'], random_state=42)

print(f"训练集大小（预期）: {len(train_files)}")
print(f"验证集大小（预期）: {len(val_files)}")
print(f"测试集大小（预期）: {len(test_files)}")

# 创建训练、验证和测试集的目录
train_dir = os.path.join(main_dir, 'train')
val_dir = os.path.join(main_dir, 'val')
test_dir = os.path.join(main_dir, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 函数：复制文件到指定目录并生成dataset.txt
def copy_files_and_create_txt(file_list, target_dir, dataset_filename):
    video_dir = target_dir
    copied_files = 0
    with open(os.path.join(main_dir, dataset_filename), 'w') as f:
        for _, row in file_list.iterrows():
            mp4_file = row['filename']
            labels = row['labels']
            src_path = os.path.join(mp4_dir, mp4_file)
            dst_path = os.path.join(video_dir, mp4_file)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                copied_files += 1
            else:
                print(f"文件不存在: {src_path}")  # 打印不存在的文件
            f.write(f"{mp4_file} {labels}\n")
    print(f"{dataset_filename} - 实际复制文件数: {copied_files}")  # 调试信息

# 复制文件到相应目录并生成dataset.txt
copy_files_and_create_txt(train_files, train_dir, 'train_dataset.txt')
copy_files_and_create_txt(val_files, val_dir, 'val_dataset.txt')
copy_files_and_create_txt(test_files, test_dir, 'test_dataset.txt')

print(f"训练集大小: {len(train_files)}")
print(f"验证集大小: {len(val_files)}")
print(f"测试集大小: {len(test_files)}")
