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

# 分开处理标签并计算每个标签的出现次数
all_labels = df['labels'].str.split('_').explode()
label_counts = all_labels.value_counts()

# 筛选出出现次数大于1的标签
valid_labels = label_counts[label_counts > 8].index

# 过滤数据框，只保留包含有效标签的行
def has_valid_label(labels, valid_labels):
    return any(label in valid_labels for label in labels.split('_'))

df = df[df['labels'].apply(has_valid_label, valid_labels=valid_labels)]

# 按照7:2:1的比例分割数据集
train_files, temp_files = train_test_split(df, test_size=0.3, stratify=df['labels'], random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=2/3, stratify=temp_files['labels'], random_state=42)

# 创建训练、验证和测试集的目录
train_dir = os.path.join(main_dir, 'train')
val_dir = os.path.join(main_dir, 'val')
test_dir = os.path.join(main_dir, 'test')
os.makedirs(os.path.join(train_dir, 'video'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'video'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'video'), exist_ok=True)

# 函数：复制文件到指定目录并生成dataset.txt
def copy_files_and_create_txt(file_list, target_dir):
    video_dir = os.path.join(target_dir, 'video')
    with open(os.path.join(target_dir, 'dataset.txt'), 'w') as f:
        for _, row in file_list.iterrows():
            mp4_file = row['filename']
            labels = row['labels']
            src_path = os.path.join(mp4_dir, mp4_file)
            dst_path = os.path.join(video_dir, mp4_file)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            f.write(f"{mp4_file} {labels}\n")

# 复制文件到相应目录并生成dataset.txt
copy_files_and_create_txt(train_files, train_dir)
copy_files_and_create_txt(val_files, val_dir)
copy_files_and_create_txt(test_files, test_dir)

print(f"训练集大小: {len(train_files)}")
print(f"验证集大小: {len(val_files)}")
print(f"测试集大小: {len(test_files)}")
