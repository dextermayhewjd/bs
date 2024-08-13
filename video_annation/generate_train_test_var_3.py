import os
import shutil
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np
# 读取并解析原始注释文件
def read_annotations(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    annotations = [line.strip() for line in lines]
    return annotations

# 提取标签
def extract_labels(annotations):
    labels = [annotation.split()[1:] for annotation in annotations]
    return labels

# 移除稀有标签
def remove_rare_labels(annotations, min_count=10):
    labels = extract_labels(annotations)
    flat_labels = [label for sublist in labels for label in sublist]
    label_counts = Counter(flat_labels)
    
    filtered_annotations = []
    for annotation in annotations:
        tags = annotation.split()[1:]
        if all(label_counts[tag] >= min_count for tag in tags):
            filtered_annotations.append(annotation)
    
    return filtered_annotations

# 检查并移除样本数少于2的标签
def remove_insufficient_labels(annotations):
    labels = extract_labels(annotations)
    flat_labels = [label for sublist in labels for label in sublist]
    label_counts = Counter(flat_labels)
    
    filtered_annotations = []
    for annotation in annotations:
        tags = annotation.split()[1:]
        if all(label_counts[tag] >= 2 for tag in tags):
            filtered_annotations.append(annotation)
    
    return filtered_annotations

# 分割数据集
def split_dataset(annotations, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    labels = extract_labels(annotations)
    # Flatten the labels for stratification
    flat_labels = [' '.join(label_list) for label_list in labels]
    train_and_val_set, test_set = train_test_split(annotations, test_size=test_ratio, stratify=flat_labels)
    train_labels = extract_labels(train_and_val_set)
    flat_train_labels = [' '.join(label_list) for label_list in train_labels]
    train_set, val_set = train_test_split(train_and_val_set, test_size=val_ratio/(train_ratio + val_ratio), stratify=flat_train_labels)
    return train_set, val_set, test_set

# 写入新的注释文件并移动文件
def write_annotations_and_move_files(annotations, file_path, destination_folder, source_folder='all'):
    os.makedirs(destination_folder, exist_ok=True)
    with open(file_path, 'w') as file:
        for annotation in annotations:
            file.write(annotation + '\n')
            mp4_file = annotation.split()[0]
            source_path = os.path.join(source_folder, mp4_file)
            destination_path = os.path.join(destination_folder, mp4_file)
            if os.path.exists(source_path):
                shutil.move(source_path, destination_path)
            else:
                print(f"File {source_path} does not exist!")
# 创建文件夹
def create_folders():
    os.makedirs('train', exist_ok=True)
    os.makedirs('val', exist_ok=True)
    os.makedirs('test', exist_ok=True)
# 主程序
def main():
    
    main_dir = r'D:\dog_video\Feb\2_2_kt\all_good'
    mp4_dir = os.path.join(main_dir, 'all')
    dataset_file = os.path.join(main_dir, 'dataset.txt')

    
    input_file = dataset_file  # 原始注释文件路径
    train_file = os.path.join(main_dir, 'train_dataset.txt')  # 训练集注释文件路径
    val_file = os.path.join(main_dir, 'val_dataset.txt')  # 验证集注释文件路径
    test_file = os.path.join(main_dir, 'test_dataset.txt')  # 测试集注释文件路径
    
    annotations = read_annotations(input_file)
    filtered_annotations = remove_rare_labels(annotations)
    filtered_annotations = remove_insufficient_labels(filtered_annotations)
    train_set, val_set, test_set = split_dataset(filtered_annotations)
    
    create_folders()
    write_annotations_and_move_files(train_set, train_file, 'train',mp4_dir)
    write_annotations_and_move_files(val_set, val_file, 'val',mp4_dir)
    write_annotations_and_move_files(test_set, test_file, 'test',mp4_dir)

# 运行主程序
if __name__ == '__main__':
    main()